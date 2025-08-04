import yfinance as yf
import pandas as pd
from skfolio.optimization import MeanRisk, ObjectiveFunction, HierarchicalRiskParity
from skfolio.measures import RiskMeasure
from skfolio.preprocessing import prices_to_returns
import matplotlib.pyplot as plt
import numpy as np
import time
import skfolio.portfolio
import curl_cffi.requests as requests
import pygwalker as pyg

# --- Configuration ---
max_retries = 3
retry_delay_seconds = 60
tickers = ["MSFT", "AAPL", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "PG", "JNJ"]
start_date = "2020-01-01"
end_date = None
annual_risk_free_rate = 0.02
daily_risk_free_rate = annual_risk_free_rate / 252
num_top_assets_selection = 5

# --- 1. Data Acquisition ---
session = requests.Session(impersonate="chrome")
data = None
for attempt in range(max_retries):
    print(f"Downloading historical data for {tickers} from {start_date} to {end_date if end_date else 'current date'}... (Attempt {attempt + 1}/{max_retries})")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, session=session)
        prices = data['Close']
        print("Data downloaded and 'Close' extracted successfully.")
        break
    except KeyError as ke:
        print(f"Error extracting 'Close' from downloaded data: {ke}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
        else:
            print("Max retries reached. Could not download data. Exiting.")
            exit()
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
        else:
            print("Max retries reached. Could not download data. Exiting.")
            exit()

if data is None or prices.empty:
    print("No data was successfully downloaded after all retries or prices DataFrame is initially empty. Exiting.")
    exit()

# --- 2. Data Preprocessing & NaN Handling ---
print("\n--- Handling Missing Price Data (NaNs) ---")
prices = prices.ffill().bfill()
prices = prices.dropna(axis=1, how='all').dropna(axis=0)
if prices.empty:
    print("No complete price data available after handling NaNs. Exiting.")
    exit()
returns = prices_to_returns(prices)

# --- 3. Split Data into Training and Testing Sets ---
train_size = int(len(returns) * 0.7)
X_train = returns.iloc[:train_size]
X_test = returns.iloc[train_size:]
if X_train.empty or X_test.empty:
    print("Not enough data to create meaningful train/test sets. Exiting.")
    exit()

# --- 4. Asset Selection for Subset Portfolio ---
print("\n--- Locating Best Subset of Performing Assets ---")
asset_sharpe_ratios = {}
for col in X_train.columns:
    asset_returns = X_train[col]
    asset_std_dev = asset_returns.std()
    if asset_std_dev > 0:
        asset_excess_returns = asset_returns - daily_risk_free_rate
        sharpe = (asset_excess_returns.mean() / asset_excess_returns.std()) * np.sqrt(252)
        asset_sharpe_ratios[col] = sharpe
    else:
        asset_sharpe_ratios[col] = -np.inf
sorted_assets = sorted(asset_sharpe_ratios.items(), key=lambda item: item[1], reverse=True)
selected_tickers = [asset[0] for asset in sorted_assets[:min(num_top_assets_selection, len(X_train.columns))]]
if not selected_tickers:
    print("No assets selected for subset. Exiting.")
    exit()
X_train_subset = X_train[selected_tickers]
X_test_subset = X_test[selected_tickers]
if X_train_subset.empty or X_test_subset.empty:
    print("Subset data is empty after selection. Exiting.")
    exit()

# --- 5. Portfolio Optimization & Evaluation ---
mean_risk_model = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.STANDARD_DEVIATION, min_weights=0.0, max_weights=1.0)
mean_risk_model.fit(X_train)
portfolio_mean_risk_test = mean_risk_model.predict(X_test)
num_assets_ew = len(X_test.columns)
equal_weights_ew = np.array([1 / num_assets_ew] * num_assets_ew)
ew_portfolio_daily_returns = X_test.dot(equal_weights_ew)
ew_portfolio_cumulative_returns = (1 + ew_portfolio_daily_returns).cumprod() - 1

def calculate_max_drawdown_manual(cumulative_returns_series):
    peak_value = cumulative_returns_series.expanding(min_periods=1).max()
    drawdown = (cumulative_returns_series - peak_value) / peak_value
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).dropna()
    if drawdown.empty:
        return 0.0
    return abs(drawdown.min())

annualized_mean_return_ew = ew_portfolio_daily_returns.mean() * 252
annualized_volatility_ew = ew_portfolio_daily_returns.std() * np.sqrt(252)
max_drawdown_ew = calculate_max_drawdown_manual(ew_portfolio_cumulative_returns)
excess_returns_daily_ew = ew_portfolio_daily_returns - daily_risk_free_rate
sharpe_ratio_ew = (excess_returns_daily_ew.mean() / excess_returns_daily_ew.std()) * np.sqrt(252)
hrp_model = HierarchicalRiskParity(min_weights=0.0, max_weights=1.0)
hrp_model.fit(X_train)
portfolio_hrp_test = hrp_model.predict(X_test)
mean_risk_subset_model = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.STANDARD_DEVIATION, min_weights=0.0, max_weights=1.0)
mean_risk_subset_model.fit(X_train_subset)
portfolio_mean_risk_subset_test = mean_risk_subset_model.predict(X_test_subset)

# --- 6. Visualize Cumulative Returns (Matplotlib) ---
print("\nPlotting cumulative returns...")
plt.figure(figsize=(14, 7))

optimized_mean_risk_cumulative_returns = portfolio_mean_risk_test.cumulative_returns
hrp_cumulative_returns = portfolio_hrp_test.cumulative_returns
subset_cumulative_returns = portfolio_mean_risk_subset_test.cumulative_returns
plot_dates = X_test.index

plt.plot(plot_dates, optimized_mean_risk_cumulative_returns, label='Optimized Portfolio (Mean-Risk - Full Assets)', color='blue')
plt.plot(plot_dates, hrp_cumulative_returns, label='Hierarchical Risk Parity (HRP - Full Assets)', color='green', linestyle='--')
plt.plot(ew_portfolio_cumulative_returns.index, ew_portfolio_cumulative_returns, label='Equal Weighted Portfolio (Full Assets)', color='orange')
plt.plot(plot_dates, subset_cumulative_returns, label=f'Optimized Subset Portfolio (Top {num_top_assets_selection} Assets)', color='purple', linestyle='-.')

plt.title('Cumulative Returns: All Portfolios vs. Optimized Subset Portfolio (Test Set)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 7. Prepare Data for PyGWalker Dashboard (Summary of Portfolio Performance) ---
print("\n--- Preparing data for PyGWalker Dashboard (Summary) ---")
portfolio_results = []
portfolio_results.append({'Portfolio Strategy': 'Mean-Risk (Full Assets)', 'Annualized Mean Return': portfolio_mean_risk_test.annualized_mean, 'Annualized Volatility': portfolio_mean_risk_test.annualized_standard_deviation, 'Annualized Sharpe Ratio': portfolio_mean_risk_test.annualized_sharpe_ratio, 'Maximum Drawdown': portfolio_mean_risk_test.max_drawdown, 'Number of Assets': len(tickers)})
portfolio_results.append({'Portfolio Strategy': 'HRP (Full Assets)', 'Annualized Mean Return': portfolio_hrp_test.annualized_mean, 'Annualized Volatility': portfolio_hrp_test.annualized_standard_deviation, 'Annualized Sharpe Ratio': portfolio_hrp_test.annualized_sharpe_ratio, 'Maximum Drawdown': portfolio_hrp_test.max_drawdown, 'Number of Assets': len(tickers)})
portfolio_results.append({'Portfolio Strategy': 'Equal Weighted (Full Assets)', 'Annualized Mean Return': annualized_mean_return_ew, 'Annualized Volatility': annualized_volatility_ew, 'Annualized Sharpe Ratio': sharpe_ratio_ew, 'Maximum Drawdown': max_drawdown_ew, 'Number of Assets': len(tickers)})
portfolio_results.append({'Portfolio Strategy': f'Mean-Risk (Top {num_top_assets_selection} Assets)', 'Annualized Mean Return': portfolio_mean_risk_subset_test.annualized_mean, 'Annualized Volatility': portfolio_mean_risk_subset_test.annualized_standard_deviation, 'Annualized Sharpe Ratio': portfolio_mean_risk_subset_test.annualized_sharpe_ratio, 'Maximum Drawdown': portfolio_mean_risk_subset_test.max_drawdown, 'Number of Assets': num_top_assets_selection})
results_df = pd.DataFrame(portfolio_results)
pyg.walk(results_df, "portfolio_summary_dashboard")

# --- NEW PYGWALKER EXAMPLES ---
# Example 1: Efficient Frontier Data
print("\n--- Preparing Efficient Frontier Data for PyGWalker ---")
ef_model = MeanRisk(
    objective_function=ObjectiveFunction.MINIMIZE_RISK,
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    min_weights=0.0,
    max_weights=1.0,
    efficient_frontier_size=100
)
try:
    ef_model.fit(X_train)
    efficient_frontier = ef_model.predict(X_train)
    if not isinstance(efficient_frontier, skfolio.portfolio.Population):
        raise TypeError("The 'predict' method did not return a Population object.")
    ef_data = {
        'Annualized Volatility': [p.annualized_standard_deviation for p in efficient_frontier],
        'Annualized Mean Return': [p.annualized_mean for p in efficient_frontier],
        'Annualized Sharpe Ratio': [p.annualized_sharpe_ratio for p in efficient_frontier],
        'Portfolio Type': 'Efficient Frontier'
    }
    efficient_frontier_df = pd.DataFrame(ef_data)
    tangency_portfolio_on_train = mean_risk_model.predict(X_train)
    tangency_data = {
        'Annualized Volatility': [tangency_portfolio_on_train.annualized_standard_deviation],
        'Annualized Mean Return': [tangency_portfolio_on_train.annualized_mean],
        'Annualized Sharpe Ratio': [tangency_portfolio_on_train.annualized_sharpe_ratio],
        'Portfolio Type': 'Max Sharpe (Tangency)'
    }
    tangency_df = pd.DataFrame(tangency_data)
    efficient_frontier_combined_df = pd.concat([efficient_frontier_df, tangency_df], ignore_index=True)
    pyg.walk(efficient_frontier_combined_df, "efficient_frontier_dashboard")
except Exception as e:
    print(f"An error occurred while generating the efficient frontier: {e}")

# Example 2: Asset Weights Distribution for different strategies
print("\n--- Preparing Asset Weights Data for PyGWalker ---")
weights_data = []
for i, asset in enumerate(X_train.columns):
    weights_data.append({'Portfolio Strategy': 'Mean-Risk (Full Assets)', 'Asset': asset, 'Weight': mean_risk_model.weights_[i]})
for i, asset in enumerate(X_train.columns):
    weights_data.append({'Portfolio Strategy': 'HRP (Full Assets)', 'Asset': asset, 'Weight': hrp_model.weights_[i]})
for i, asset in enumerate(selected_tickers):
    weights_data.append({'Portfolio Strategy': f'Mean-Risk (Top {num_top_assets_selection} Assets)', 'Asset': asset, 'Weight': mean_risk_subset_model.weights_[i]})
ew_full_weights = np.array([1 / len(X_train.columns)] * len(X_train.columns))
for i, asset in enumerate(X_train.columns):
    weights_data.append({'Portfolio Strategy': 'Equal Weighted (Full Assets)', 'Asset': asset, 'Weight': ew_full_weights[i]})
weights_df = pd.DataFrame(weights_data)
pyg.walk(weights_df, "asset_weights_dashboard")


# Example 3: Risk Contributions (for chosen strategies)
print("\n--- Preparing Risk Contributions Data for PyGWalker (Revised and Corrected) ---")
risk_contribution_data = []

# Manual Calculation of Risk Contributions for HRP
if len(hrp_model.weights_) == X_train.shape[1]:
    hrp_weights = hrp_model.weights_
    hrp_covariance = X_train.cov().values
    hrp_portfolio_std = np.sqrt(hrp_weights.T @ hrp_covariance @ hrp_weights)
    if hrp_portfolio_std > 0:
        hrp_mcr = (hrp_covariance @ hrp_weights) / hrp_portfolio_std
        hrp_rc = hrp_weights * hrp_mcr
        for i, asset in enumerate(X_train.columns):
            risk_contribution_data.append({'Asset': asset, 'Risk Contribution': hrp_rc[i], 'Portfolio Strategy': 'HRP (Full Assets)'})
    else:
        print("Cannot calculate HRP risk contributions: portfolio standard deviation is zero.")
else:
    print("Cannot calculate HRP risk contributions: weights dimensions do not match data.")


# Manual Calculation of Risk Contributions for Mean-Risk
if len(mean_risk_model.weights_) == X_train.shape[1]:
    mr_weights = mean_risk_model.weights_
    mr_covariance = X_train.cov().values
    mr_portfolio_std = np.sqrt(mr_weights.T @ mr_covariance @ mr_weights)
    if mr_portfolio_std > 0:
        mr_mcr = (mr_covariance @ mr_weights) / mr_portfolio_std
        mr_rc = mr_weights * mr_mcr
        for i, asset in enumerate(X_train.columns):
            risk_contribution_data.append({'Asset': asset, 'Risk Contribution': mr_rc[i], 'Portfolio Strategy': 'Mean-Risk (Full Assets)'})
    else:
        print("Cannot calculate Mean-Risk risk contributions: portfolio standard deviation is zero.")
else:
    print("Cannot calculate Mean-Risk risk contributions: weights dimensions do not match data.")


risk_contribution_df = pd.DataFrame(risk_contribution_data)
if not risk_contribution_df.empty:
    pyg.walk(risk_contribution_df, "risk_contributions_dashboard")
else:
    print("No risk contribution data generated for PyGWalker.")


# Example 4: Rolling Performance Metrics
print("\n--- Preparing Rolling Performance Data for PyGWalker (Revised and Corrected) ---")
optimized_mean_risk_daily_returns_series = pd.Series(
    data=portfolio_mean_risk_test.returns,
    index=X_test.index
)
window_size = 60
annualization_factor = np.sqrt(252)

rolling_mean_return = optimized_mean_risk_daily_returns_series.rolling(window=window_size).mean()
rolling_std_dev = optimized_mean_risk_daily_returns_series.rolling(window=window_size).std()

rolling_sharpe = (rolling_mean_return - daily_risk_free_rate) / rolling_std_dev
rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

rolling_df = pd.DataFrame({
    'Date': rolling_sharpe.index,
    'Rolling Sharpe Ratio (60-Day)': rolling_sharpe.values * annualization_factor,
    'Rolling Volatility (60-Day)': rolling_std_dev.values * annualization_factor,
    'Rolling Return (60-Day)': rolling_mean_return.values * 252,
    'Portfolio Strategy': 'Mean-Risk (Full Assets)'
}).dropna()

if not rolling_df.empty:
    pyg.walk(rolling_df, "rolling_performance_dashboard")
else:
    print("No rolling performance data generated for PyGWalker.")


# Example 5: Individual Asset Performance on Test Set
print("\n--- Preparing Individual Asset Performance Data for PyGWalker ---")
individual_asset_performance_data = []
for col in X_test.columns:
    asset_returns = X_test[col]
    asset_mean_return = asset_returns.mean() * 252
    asset_std_dev = asset_returns.std() * np.sqrt(252)
    if asset_std_dev > 0:
        asset_excess_returns_daily = asset_returns - daily_risk_free_rate
        asset_sharpe = (asset_excess_returns_daily.mean() / asset_excess_returns_daily.std()) * np.sqrt(252)
    else:
        asset_sharpe = 0
    asset_cumulative_returns = (1 + asset_returns).cumprod() - 1
    asset_max_drawdown = calculate_max_drawdown_manual(asset_cumulative_returns)
    individual_asset_performance_data.append({
        'Asset': col,
        'Annualized Mean Return': asset_mean_return,
        'Annualized Volatility': asset_std_dev,
        'Annualized Sharpe Ratio': asset_sharpe,
        'Maximum Drawdown': asset_max_drawdown
    })
individual_asset_perf_df = pd.DataFrame(individual_asset_performance_data)
pyg.walk(individual_asset_perf_df, "individual_asset_performance_dashboard")


# Example 6: Cumulative Returns Comparison (Normalized)
print("\n--- Preparing Normalized Cumulative Returns Data for PyGWalker ---")
cumulative_returns_data = []
df_temp = pd.DataFrame({'Date': X_test.index, 'Cumulative Return': (1 + portfolio_mean_risk_test.cumulative_returns), 'Portfolio Strategy': 'Mean-Risk (Full Assets)'})
cumulative_returns_data.append(df_temp)
df_temp = pd.DataFrame({'Date': X_test.index, 'Cumulative Return': (1 + portfolio_hrp_test.cumulative_returns), 'Portfolio Strategy': 'HRP (Full Assets)'})
cumulative_returns_data.append(df_temp)
df_temp = pd.DataFrame({'Date': ew_portfolio_cumulative_returns.index, 'Cumulative Return': (1 + ew_portfolio_cumulative_returns).values, 'Portfolio Strategy': 'Equal Weighted (Full Assets)'})
cumulative_returns_data.append(df_temp)
df_temp = pd.DataFrame({'Date': X_test_subset.index, 'Cumulative Return': (1 + portfolio_mean_risk_subset_test.cumulative_returns), 'Portfolio Strategy': f'Mean-Risk (Top {num_top_assets_selection} Assets)'})
cumulative_returns_data.append(df_temp)
cumulative_returns_combined_df = pd.concat(cumulative_returns_data, ignore_index=True)
pyg.walk(cumulative_returns_combined_df, "cumulative_returns_dashboard")