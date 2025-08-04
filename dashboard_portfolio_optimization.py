import yfinance as yf
import pandas as pd
from skfolio.optimization import MeanRisk, ObjectiveFunction, HierarchicalRiskParity
from skfolio.measures import RiskMeasure
from skfolio.preprocessing import prices_to_returns
import matplotlib.pyplot as plt
import numpy as np
import time
import skfolio.portfolio # Ensure this is imported for skfolio portfolio objects

# Import requests from curl_cffi to create a session
import curl_cffi.requests as requests
import pygwalker as pyg

max_retries = 3
retry_delay_seconds = 60

tickers = ["MSFT", "AAPL", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "PG", "JNJ"]
start_date = "2020-01-01"
end_date = None # This requests data up to the current date.

session = requests.Session(impersonate="chrome")

print(f"Attempting to download historical data for {tickers} from {start_date} to {end_date if end_date else 'current date'}...")

data = None

for attempt in range(max_retries):
    print(f"Downloading historical data for {tickers} from {start_date} to {end_date if end_date else 'current date'}... (Attempt {attempt + 1}/{max_retries})")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, session=session)
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

        prices = data['Close']
        print("Data downloaded and 'Close' extracted successfully.")

        print("\n--- CRITICAL DEBUG: Data and Prices State AFTER 'Close' Extraction ---")
        print(f"Is 'data' DataFrame empty? {data.empty}")
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                print("'Close' is in the first level of MultiIndex columns.")
            else:
                print(f"'Close' is NOT in the first level of MultiIndex columns. Available first levels: {data.columns.levels[0]}")
        elif 'Close' not in data.columns:
            print(f"'Close' column not found in 'data' DataFrame. Available columns: {data.columns}")

        print(f"Shape of 'prices' DataFrame after data['Close']: {prices.shape}")
        print(f"Is 'prices' DataFrame empty? {prices.empty}")
        if not prices.empty:
            print("Head of 'prices' DataFrame after data['Close']:")
            print(prices.head())
        else:
            print("!!! 'prices' DataFrame is EMPTY immediately after data['Close'] extraction. !!!")
        print("!!! This can happen if all tickers failed to download or 'Close' is not the primary column. !!!")
        print("--- CRITICAL DEBUG END ---\n")

        break
    except KeyError as ke:
        print(f"Error extracting 'Close' from downloaded data: {ke}")
        print("This likely means data was not downloaded correctly or 'Close' column is missing.")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
        else:
            print("Max retries reached. Could not download data. Exiting.")
            print("Please check tickers and date range, or try again later.")
            exit()
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
        else:
            print("Max retries reached. Could not download data. Exiting.")
            print("Please check tickers and date range, or try again later.")
            exit()

if data is None or prices.empty:
    print("No data was successfully downloaded after all retries or prices DataFrame is initially empty. Exiting.")
    exit()

# --- Handle potential missing data by filling instead of dropping ---
print("\n--- Handling Missing Price Data (NaNs) ---")
print(f"Shape of 'prices' BEFORE NaN handling: {prices.shape}")
print(f"Number of NaNs per column BEFORE NaN handling:\n{prices.isna().sum()}")

# 1. Forward-fill missing values: propagate last valid observation forward
prices = prices.fillna(method='ffill')

# 2. Backward-fill any remaining missing values (e.g., NaNs at the very beginning of the series)
prices = prices.fillna(method='bfill')

# 3. After filling, check if any columns are *still* all NaN (meaning they had no data at all)
initial_cols = prices.shape[1]
prices = prices.dropna(axis=1, how='all') # Drop columns that are entirely NaN after filling
if prices.shape[1] < initial_cols:
    print(f"Dropped {initial_cols - prices.shape[1]} columns that were entirely NaN after filling.")

# 4. Drop any rows (dates) that might still contain NaNs (less common after ffill/bfill for financial data)
initial_rows = prices.shape[0]
prices = prices.dropna(axis=0)
if prices.shape[0] < initial_rows:
    print(f"Dropped {initial_rows - prices.shape[0]} rows (dates) due to remaining NaNs.")

if prices.empty:
    print("No complete price data available after handling NaNs. Exiting.")
    exit()

print(f"Shape of 'prices' AFTER NaN handling: {prices.shape}")
print(f"Number of NaNs per column AFTER NaN handling:\n{prices.isna().sum()}")
print("Sample of prices data after NaN handling:")
print(prices.head())
# --- END MODIFIED NaN HANDLING ---


# 3. Convert prices to returns
returns = prices_to_returns(prices)
print("\nSample of returns data:")
print(returns.head())

# Define annual risk-free rate early so it's available for asset selection and Sharpe Ratio calculation
annual_risk_free_rate = 0.02 # Example: 2% annual risk-free rate
daily_risk_free_rate = annual_risk_free_rate / 252 # Convert to daily rate for daily returns

# 4. Split data into training and testing sets
train_size = int(len(returns) * 0.7)
X_train = returns.iloc[:train_size]
X_test = returns.iloc[train_size:]

print(f"\nTraining set size: {len(X_train)} observations")
print(f"Testing set size: {len(X_test)} observations")

if X_train.empty or X_test.empty:
    print("Not enough data to create meaningful train/test sets. Please extend your date range.")
    exit()

# --- NEW: Locate Best Subset of Performing Assets ---
print("\n--- Locating Best Subset of Performing Assets ---")

# Calculate individual asset annualized Sharpe Ratios on the training data
asset_sharpe_ratios = {}

for col in X_train.columns:
    asset_returns = X_train[col]
    asset_std_dev = asset_returns.std()

    if asset_std_dev > 0: # Avoid division by zero
        # Daily excess return
        asset_excess_returns = asset_returns - daily_risk_free_rate
        # Annualized Sharpe Ratio
        sharpe = (asset_excess_returns.mean() / asset_excess_returns.std()) * np.sqrt(252)
        asset_sharpe_ratios[col] = sharpe
    else:
        # Assign a very low value if std dev is zero (means no volatility, which is too good to be true or an issue)
        # or if the asset has no valid data in the training period (after NaNs handled)
        asset_sharpe_ratios[col] = -np.inf

# Sort assets by Sharpe Ratio in descending order
sorted_assets = sorted(asset_sharpe_ratios.items(), key=lambda item: item[1], reverse=True)

# Select the top N assets (e.g., top 5). Use min() to ensure we don't try to select more assets than available.
num_top_assets = min(5, len(X_train.columns))
selected_tickers = [asset[0] for asset in sorted_assets[:num_top_assets]]

print(f"Top {num_top_assets} performing assets based on training Sharpe Ratio:")
for ticker, sharpe in sorted_assets[:num_top_assets]:
    print(f"- {ticker}: Sharpe Ratio = {sharpe:.4f}")

if not selected_tickers:
    print("No assets selected for subset. Exiting.")
    exit()

# Filter training and testing data for the selected subset
X_train_subset = X_train[selected_tickers]
X_test_subset = X_test[selected_tickers]

print(f"\nTraining subset size: {len(X_train_subset)} observations, {len(selected_tickers)} assets")
print(f"Testing subset size: {len(X_test_subset)} observations, {len(selected_tickers)} assets")

if X_train_subset.empty or X_test_subset.empty:
    print("Subset data is empty after selection. This might happen if all selected assets were dropped due to NaNs or had no valid data. Exiting.")
    exit()
# --- END NEW: Locate Best Subset of Performing Assets ---


# 5. Define and fit a skfolio portfolio optimization model (Mean-Risk - Max Sharpe) for FULL assets
mean_risk_model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    min_weights=0.0, # No short selling
    max_weights=1.0, # Max 100% in any asset
    efficient_frontier_size=None # We only want the optimal portfolio, not the full frontier
)

print("\nFitting the Mean-Risk model (Full Assets) on training data...")
mean_risk_model.fit(X_train)
print("Mean-Risk Model (Full Assets) fitted successfully.")

# Print the optimized weights
print("\nMean-Risk Optimized Portfolio Weights (Full Assets, from training data):")
mean_risk_weights_df = pd.DataFrame([mean_risk_model.weights_], columns=X_train.columns, index=['Weights']).T
print(mean_risk_weights_df)
print(f"Sum of weights: {mean_risk_weights_df['Weights'].sum():.4f}")


# 6. Evaluate the Mean-Risk optimized portfolio (Full Assets) on the test set
print("\nEvaluating the Mean-Risk optimized portfolio (Full Assets) on the test set...")
portfolio_mean_risk_test = mean_risk_model.predict(X_test)

print("\nMean-Risk Optimized Portfolio Performance (Full Assets) on Test Set:")
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
ew_portfolio_daily_returns = X_test.dot(equal_weights_ew)

# Calculate cumulative returns for plotting
ew_portfolio_cumulative_returns = (1 + ew_portfolio_daily_returns).cumprod() - 1

# Define a helper function to calculate Maximum Drawdown
def calculate_max_drawdown_manual(cumulative_returns_series):
    peak_value = cumulative_returns_series.expanding(min_periods=1).max()
    drawdown = (cumulative_returns_series - peak_value) / peak_value
    # Handle cases where peak_value might be zero or negative, leading to inf drawdown
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan)
    max_drawdown = drawdown.min()
    return max_drawdown

# Calculate performance metrics for the Equal Weighted portfolio
annualized_mean_return_ew = ew_portfolio_daily_returns.mean() * 252
annualized_volatility_ew = ew_portfolio_daily_returns.std() * np.sqrt(252)
max_drawdown_ew = calculate_max_drawdown_manual(ew_portfolio_cumulative_returns)

# Annualized Sharpe Ratio (uses the already defined annual_risk_free_rate and daily_risk_free_rate)
excess_returns_daily_ew = ew_portfolio_daily_returns - daily_risk_free_rate
sharpe_ratio_ew = (excess_returns_daily_ew.mean() / excess_returns_daily_ew.std()) * np.sqrt(252)

print("\nEqual Weighted Portfolio Performance (Manual Calculation) on Test Set:")
print(f"Annualized Mean Return: {annualized_mean_return_ew:.4f}")
print(f"Annualized Volatility: {annualized_volatility_ew:.4f}")
print(f"Annualized Sharpe Ratio: {sharpe_ratio_ew:.4f}")
print(f"Maximum Drawdown: {max_drawdown_ew:.4f}")
# --- END MANUAL EQUAL-WEIGHTED PORTFOLIO CALCULATION ---

# --- HIERARCHICAL RISK PARITY (HRP) MODEL (Full Assets) ---
print("\n--- HIERARCHICAL RISK PARITY (HRP) MODEL (Full Assets) ---")
hrp_model = HierarchicalRiskParity(
    min_weights=0.0, # No short selling
    max_weights=1.0 # Max 100% in any asset
)

print("\nFitting the HRP model (Full Assets) on training data...")
hrp_model.fit(X_train)
print("HRP Model (Full Assets) fitted successfully.")

# Print the HRP optimized weights
print("\nHRP Optimized Portfolio Weights (Full Assets, from training data):")
hrp_weights_df = pd.DataFrame([hrp_model.weights_], columns=X_train.columns, index=['Weights']).T
print(hrp_weights_df)
print(f"Sum of weights: {hrp_weights_df['Weights'].sum():.4f}")

# Evaluate the HRP optimized portfolio on the test set
print("\nEvaluating the HRP optimized portfolio (Full Assets) on the test set...")
portfolio_hrp_test = hrp_model.predict(X_test)

print("\nHRP Optimized Portfolio Performance (Full Assets) on Test Set:")
print(f"Annualized Mean Return: {portfolio_hrp_test.annualized_mean:.4f}")
print(f"Annualized Volatility: {portfolio_hrp_test.annualized_standard_deviation:.4f}")
print(f"Annualized Sharpe Ratio: {portfolio_hrp_test.annualized_sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {portfolio_hrp_test.max_drawdown:.4f}")
# --- END HIERARCHICAL RISK PARITY (HRP) MODEL ---


# --- NEW: Define and fit a skfolio portfolio optimization model for the SUBSET ---
print("\n--- OPTIMIZING SUBSET PORTFOLIO (Mean-Risk - Max Sharpe) ---")
mean_risk_subset_model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    min_weights=0.0, # No short selling
    max_weights=1.0, # Max 100% in any asset
    efficient_frontier_size=None # We only want the optimal portfolio
)
print("\nFitting a Mean-Risk model on the SUBSET training data...")
mean_risk_subset_model.fit(X_train_subset)
print("Mean-Risk Subset Model fitted successfully.")

print("\nMean-Risk Optimized SUBSET Portfolio Weights (from training data):")
mean_risk_subset_weights_df = pd.DataFrame([mean_risk_subset_model.weights_], columns=X_train_subset.columns, index=['Weights']).T
print(mean_risk_subset_weights_df)
print(f"Sum of subset weights: {mean_risk_subset_weights_df['Weights'].sum():.4f}")

# Evaluate the Mean-Risk optimized SUBSET portfolio on the test set
print("\nEvaluating the Mean-Risk optimized SUBSET portfolio on the test set...")
portfolio_mean_risk_subset_test = mean_risk_subset_model.predict(X_test_subset)

print("\nMean-Risk Optimized SUBSET Portfolio Performance on Test Set:")
print(f"Annualized Mean Return: {portfolio_mean_risk_subset_test.annualized_mean:.4f}")
print(f"Annualized Volatility: {portfolio_mean_risk_subset_test.annualized_standard_deviation:.4f}")
print(f"Annualized Sharpe Ratio: {portfolio_mean_risk_subset_test.annualized_sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {portfolio_mean_risk_subset_test.max_drawdown:.4f}")
# --- END NEW: SUBSET PORTFOLIO OPTIMIZATION ---


# 7. Visualize Cumulative Returns (Updated to include Subset Portfolio)
print("\nPlotting cumulative returns...")
plt.figure(figsize=(14, 7)) # Slightly larger figure for more lines

optimized_mean_risk_cumulative_returns = portfolio_mean_risk_test.cumulative_returns
hrp_cumulative_returns = portfolio_hrp_test.cumulative_returns
subset_cumulative_returns = portfolio_mean_risk_subset_test.cumulative_returns # Get subset cumulative returns

# Ensure all cumulative returns align with X_test's index for plotting
plot_dates = X_test.index

plt.plot(plot_dates, optimized_mean_risk_cumulative_returns, label='Optimized Portfolio (Mean-Risk - Full Assets)', color='blue')
plt.plot(plot_dates, hrp_cumulative_returns, label='Hierarchical Risk Parity (HRP - Full Assets)', color='green', linestyle='--')
plt.plot(ew_portfolio_cumulative_returns.index, ew_portfolio_cumulative_returns, label='Equal Weighted Portfolio (Full Assets)', color='orange')
plt.plot(plot_dates, subset_cumulative_returns, label=f'Optimized Subset Portfolio (Top {num_top_assets} Assets)', color='purple', linestyle='-.') # New line for subset

plt.title('Cumulative Returns: All Portfolios vs. Optimized Subset Portfolio (Test Set)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nDetailed Summary of Mean-Risk Optimized Portfolio (Full Assets) on Test Set:")
print(portfolio_mean_risk_test.summary())

print("\nDetailed Summary of HRP Optimized Portfolio (Full Assets) on Test Set:")
print(portfolio_hrp_test.summary())

print("\nDetailed Summary of Mean-Risk Optimized SUBSET Portfolio on Test Set:")
print(portfolio_mean_risk_subset_test.summary())

# --- Prepare data for PyGWalker Dashboard ---
print("\n--- Preparing data for PyGWalker Dashboard ---")

# Create a list of dictionaries, one for each portfolio
portfolio_results = []

# Mean-Risk Full Assets
portfolio_results.append({
    'Portfolio Strategy': 'Mean-Risk (Full Assets)',
    'Annualized Mean Return': portfolio_mean_risk_test.annualized_mean,
    'Annualized Volatility': portfolio_mean_risk_test.annualized_standard_deviation,
    'Annualized Sharpe Ratio': portfolio_mean_risk_test.annualized_sharpe_ratio,
    'Maximum Drawdown': portfolio_mean_risk_test.max_drawdown,
    'Number of Assets': len(tickers)
})

# HRP Full Assets
portfolio_results.append({
    'Portfolio Strategy': 'HRP (Full Assets)',
    'Annualized Mean Return': portfolio_hrp_test.annualized_mean,
    'Annualized Volatility': portfolio_hrp_test.annualized_standard_deviation,
    'Annualized Sharpe Ratio': portfolio_hrp_test.annualized_sharpe_ratio,
    'Maximum Drawdown': portfolio_hrp_test.max_drawdown,
    'Number of Assets': len(tickers)
})

# Equal Weighted Full Assets (manual calculations)
portfolio_results.append({
    'Portfolio Strategy': 'Equal Weighted (Full Assets)',
    'Annualized Mean Return': annualized_mean_return_ew,
    'Annualized Volatility': annualized_volatility_ew,
    'Annualized Sharpe Ratio': sharpe_ratio_ew,
    'Maximum Drawdown': max_drawdown_ew,
    'Number of Assets': len(tickers)
})

# Mean-Risk Subset Assets
portfolio_results.append({
    'Portfolio Strategy': f'Mean-Risk (Top {num_top_assets} Assets)',
    'Annualized Mean Return': portfolio_mean_risk_subset_test.annualized_mean,
    'Annualized Volatility': portfolio_mean_risk_subset_test.annualized_standard_deviation,
    'Annualized Sharpe Ratio': portfolio_mean_risk_subset_test.annualized_sharpe_ratio,
    'Maximum Drawdown': portfolio_mean_risk_subset_test.max_drawdown,
    'Number of Assets': num_top_assets
})

# Create a DataFrame from the collected results
results_df = pd.DataFrame(portfolio_results)

print("\nPortfolio Performance Summary for PyGWalker:")
print(results_df)

# Launch PyGWalker for interactive dashboard
print("\nLaunching PyGWalker for interactive dashboard...")
walker = pyg.walk(results_df)