import yfinance as yf
import pandas as pd
from skfolio.optimization import MeanRisk, ObjectiveFunction, HierarchicalRiskParity # Import HierarchicalRiskParity
from skfolio.measures import RiskMeasure
from skfolio.preprocessing import prices_to_returns
import matplotlib.pyplot as plt
import numpy as np
import time
import skfolio.portfolio # Required for isinstance check and optimized portfolio plotting

# Import requests from curl_cffi to create a session
import curl_cffi.requests as requests

max_retries = 3
retry_delay_seconds = 60 # Wait 1 minute between retries

# 1. Define your assets and date range
tickers = ["MSFT", "AAPL", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "PG", "JNJ"]
start_date = "2020-01-01"
end_date = "2024-12-31" # You can adjust this to present or a past date

# Create a session to pass to yfinance, impersonating Chrome
session = requests.Session(impersonate="chrome")

# 2. Download historical data from Yahoo Finance
print(f"Attempting to download historical data for {tickers} from {start_date} to {end_date}...")

data = None # Initialize data to None

for attempt in range(max_retries):
    print(f"Downloading historical data for {tickers} from {start_date} to {end_date}... (Attempt {attempt + 1}/{max_retries})")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, session=session)
        # --- DEBUGGING START ---
        print("\n--- DEBUGGING yfinance output ---")
        print(f"Type of data: {type(data)}")
        if isinstance(data, pd.DataFrame):
            print(f"Shape of data DataFrame: {data.shape}")
            print(f"Columns of data DataFrame: {data.columns}")
            if isinstance(data.columns, pd.MultiIndex):
                print(f"MultiIndex levels: {data.columns.levels}")
            print("Head of data DataFrame:")
            print(data.head())
        else:
            print(f"Data is not a DataFrame, it's: {data}")
        print("--- DEBUGGING END ---\n")

        # Now try to access 'Close' (which is the adjusted close price)
        # Assuming auto_adjust=True (default in newer yfinance), 'Close' holds the adjusted values.
        prices = data['Close']
        print("Data downloaded and 'Close' extracted successfully.")
        break # If successful, break out of the retry loop
    except KeyError as ke: # Specific catch for KeyError if 'Close' is missing
        print(f"Error extracting 'Close' from downloaded data: {ke}")
        print("This likely means data was not downloaded correctly or 'Close' column is missing.")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
        else:
            print("Max retries reached. Could not download data. Exiting.")
            print("Please check tickers and date range, or try again later.")
            exit()
    except Exception as e: # Catch any other general exceptions
        print(f"An unexpected error occurred during download: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
        else:
            print("Max retries reached. Could not download data. Exiting.")
            print("Please check tickers and date range, or try again later.")
            exit()

# After the loop, check if data was successfully obtained
if data is None or prices.empty: # prices.empty is crucial here
    print("No data was successfully downloaded after all retries or prices DataFrame is empty. Exiting.")
    exit()

# Handle potential missing data (e.g., if a ticker didn't exist for the whole period)
prices = prices.dropna(axis=1) # Drop columns (assets) with any missing values
if prices.empty:
    print("No complete price data available for the given tickers and date range after dropping NaNs. Exiting.")
    exit()

# 3. Convert prices to returns
returns = prices_to_returns(prices)
print("\nSample of returns data:")
print(returns.head())

# 4. Split data into training and testing sets
# It's crucial to use a time-series split to avoid data leakage
train_size = int(len(returns) * 0.7)
X_train = returns.iloc[:train_size]
X_test = returns.iloc[train_size:]

print(f"\nTraining set size: {len(X_train)} observations")
print(f"Testing set size: {len(X_test)} observations")

# Ensure both train and test sets have data
if X_train.empty or X_test.empty:
    print("Not enough data to create meaningful train/test sets. Please extend your date range.")
    exit()

# 5. Define and fit a skfolio portfolio optimization model (Mean-Risk)
# Let's try to find the Maximum Sharpe Ratio portfolio
mean_risk_model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    min_weights=0.0, # No short selling
    max_weights=1.0, # Max 100% in any asset
    efficient_frontier_size=None # We only want the optimal portfolio, not the full frontier
)

print("\nFitting the Mean-Risk model on training data...")
mean_risk_model.fit(X_train)
print("Mean-Risk Model fitted successfully.")

# Print the optimized weights
print("\nMean-Risk Optimized Portfolio Weights (from training data):")
mean_risk_weights_df = pd.DataFrame([mean_risk_model.weights_], columns=X_train.columns, index=['Weights']).T
print(mean_risk_weights_df)
print(f"Sum of weights: {mean_risk_weights_df['Weights'].sum():.4f}")


# 6. Evaluate the Mean-Risk optimized portfolio on the test set
print("\nEvaluating the Mean-Risk optimized portfolio on the test set...")
portfolio_mean_risk_test = mean_risk_model.predict(X_test)

print("\nMean-Risk Optimized Portfolio Performance on Test Set:")
print(f"Annualized Mean Return: {portfolio_mean_risk_test.annualized_mean:.4f}")
print(f"Annualized Volatility: {portfolio_mean_risk_test.annualized_standard_deviation:.4f}")
print(f"Annualized Sharpe Ratio: {portfolio_mean_risk_test.annualized_sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {portfolio_mean_risk_test.max_drawdown:.4f}")

# --- MANUAL EQUAL-WEIGHTED PORTFOLIO CALCULATION (WITHOUT SKFOLIO.NAIVE) ---
print("\nCalculating Equal Weighted Portfolio performance manually...")

# Calculate equal weights for the assets in the test set
num_assets_ew = len(X_test.columns)
equal_weights_ew = np.array([1 / num_assets_ew] * num_assets_ew)

# Calculate daily returns of the Equal Weighted portfolio
# The .dot() method performs matrix multiplication: (N_days, N_assets) . (N_assets,) -> (N_days,)
ew_portfolio_daily_returns = X_test.dot(equal_weights_ew)

# Calculate cumulative returns for plotting
# (1 + r1)(1 + r2)... - 1
ew_portfolio_cumulative_returns = (1 + ew_portfolio_daily_returns).cumprod() - 1

# Define a helper function to calculate Maximum Drawdown
def calculate_max_drawdown_manual(cumulative_returns_series):
    # Calculate the running maximum value (peak)
    peak_value = cumulative_returns_series.expanding(min_periods=1).max()
    # Calculate the drawdown from the peak
    drawdown = (cumulative_returns_series - peak_value) / peak_value
    # Max drawdown is the minimum (most negative) drawdown
    max_drawdown = drawdown.min()
    return max_drawdown

# Calculate performance metrics for the Equal Weighted portfolio
# Annualized Mean Return: average daily return * number of trading days
annualized_mean_return_ew = ew_portfolio_daily_returns.mean() * 252

# Annualized Volatility: standard deviation of daily returns * sqrt(number of trading days)
annualized_volatility_ew = ew_portfolio_daily_returns.std() * np.sqrt(252)

# Maximum Drawdown
max_drawdown_ew = calculate_max_drawdown_manual(ew_portfolio_cumulative_returns)

# Annualized Sharpe Ratio (requires a risk-free rate)
# Adjust this annual risk-free rate as needed for your analysis (e.g., from FRED data)
annual_risk_free_rate = 0.02 # Example: 2% annual risk-free rate
daily_risk_free_rate = annual_risk_free_rate / 252 # Convert to daily rate for daily returns

# Calculate daily excess returns
excess_returns_daily_ew = ew_portfolio_daily_returns - daily_risk_free_rate

# Sharpe Ratio = (Mean Excess Return / Std Dev of Excess Return) * sqrt(annualization factor)
sharpe_ratio_ew = (excess_returns_daily_ew.mean() / excess_returns_daily_ew.std()) * np.sqrt(252)

print("\nEqual Weighted Portfolio Performance (Manual Calculation) on Test Set:")
print(f"Annualized Mean Return: {annualized_mean_return_ew:.4f}")
print(f"Annualized Volatility: {annualized_volatility_ew:.4f}")
print(f"Annualized Sharpe Ratio: {sharpe_ratio_ew:.4f}")
print(f"Maximum Drawdown: {max_drawdown_ew:.4f}")
# --- END MANUAL EQUAL-WEIGHTED PORTFOLIO CALCULATION ---

# --- HIERARCHICAL RISK PARITY (HRP) MODEL ---
print("\n--- HIERARCHICAL RISK PARITY (HRP) MODEL ---")
hrp_model = HierarchicalRiskParity(
    min_weights=0.0, # No short selling
    max_weights=1.0 # Max 100% in any asset
)

print("\nFitting the HRP model on training data...")
hrp_model.fit(X_train)
print("HRP Model fitted successfully.")

# Print the HRP optimized weights
print("\nHRP Optimized Portfolio Weights (from training data):")
hrp_weights_df = pd.DataFrame([hrp_model.weights_], columns=X_train.columns, index=['Weights']).T
print(hrp_weights_df)
print(f"Sum of weights: {hrp_weights_df['Weights'].sum():.4f}")

# Evaluate the HRP optimized portfolio on the test set
print("\nEvaluating the HRP optimized portfolio on the test set...")
portfolio_hrp_test = hrp_model.predict(X_test)

print("\nHRP Optimized Portfolio Performance on Test Set:")
print(f"Annualized Mean Return: {portfolio_hrp_test.annualized_mean:.4f}")
print(f"Annualized Volatility: {portfolio_hrp_test.annualized_standard_deviation:.4f}")
print(f"Annualized Sharpe Ratio: {portfolio_hrp_test.annualized_sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {portfolio_hrp_test.max_drawdown:.4f}")
# --- END HIERARCHICAL RISK PARITY (HRP) MODEL ---


# 7. Visualize Cumulative Returns
print("\nPlotting cumulative returns...")
plt.figure(figsize=(12, 6))

# Get cumulative returns data directly from the optimized portfolio object
optimized_cumulative_returns = portfolio_mean_risk_test.cumulative_returns # This is a NumPy array
hrp_cumulative_returns = portfolio_hrp_test.cumulative_returns # HRP cumulative returns

# Get the dates for the x-axis from the test set's index
# All cumulative returns align with X_test's index
plot_dates = X_test.index

# Plot Optimized Portfolio manually (Mean-Risk)
plt.plot(plot_dates, optimized_cumulative_returns, label='Optimized Portfolio (Mean-Risk - Max Sharpe)', color='blue')

# Plot HRP Portfolio
plt.plot(plot_dates, hrp_cumulative_returns, label='Hierarchical Risk Parity (HRP)', color='green', linestyle='--')

# Plot Manually Calculated Equal Weighted Portfolio
# ew_portfolio_cumulative_returns should already be a Pandas Series, so its .index is fine
plt.plot(ew_portfolio_cumulative_returns.index, ew_portfolio_cumulative_returns, label='Equal Weighted Portfolio (Manual)', color='orange')

plt.title('Cumulative Returns: Mean-Risk vs. HRP vs. Equal Weighted Portfolio (Test Set)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend() # Let matplotlib pick up the labels from the 'label' argument of each plt.plot() call
plt.grid(True)
plt.tight_layout()
plt.show()

# You can also get a detailed summary of the optimized portfolio
print("\nDetailed Summary of Mean-Risk Optimized Portfolio on Test Set:")
print(portfolio_mean_risk_test.summary())

print("\nDetailed Summary of HRP Optimized Portfolio on Test Set:")
print(portfolio_hrp_test.summary())