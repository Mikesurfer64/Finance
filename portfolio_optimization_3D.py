"""portfolio_optimization_3D.py"""

import yfinance as yf
import pandas as pd
from skfolio.optimization import MeanRisk, ObjectiveFunction, HierarchicalRiskParity
from skfolio.preprocessing import prices_to_returns
import numpy as np
import time
import skfolio.portfolio
import curl_cffi.requests as requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
mean_risk_model = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, min_weights=0.0, max_weights=1.0)
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
mean_risk_subset_model = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, min_weights=0.0, max_weights=1.0)
mean_risk_subset_model.fit(X_train_subset)
portfolio_mean_risk_subset_test = mean_risk_subset_model.predict(X_test_subset)

# ==============================================================================
# Plotly Visualizations
# ==============================================================================

# --- Plotly Chart 1: Portfolio Summary (Bar Chart) ---
print("\n--- Generating Plotly Chart 1: Portfolio Summary ---")
portfolio_results = [
    {'Portfolio Strategy': 'Mean-Risk (Full Assets)', 'Annualized Sharpe Ratio': portfolio_mean_risk_test.annualized_sharpe_ratio},
    {'Portfolio Strategy': 'HRP (Full Assets)', 'Annualized Sharpe Ratio': portfolio_hrp_test.annualized_sharpe_ratio},
    {'Portfolio Strategy': 'Equal Weighted (Full Assets)', 'Annualized Sharpe Ratio': sharpe_ratio_ew},
    {'Portfolio Strategy': f'Mean-Risk (Top {num_top_assets_selection} Assets)', 'Annualized Sharpe Ratio': portfolio_mean_risk_subset_test.annualized_sharpe_ratio}
]
results_df = pd.DataFrame(portfolio_results)
fig = px.bar(results_df, x='Portfolio Strategy', y='Annualized Sharpe Ratio',
             title='Portfolio Performance: Annualized Sharpe Ratio Comparison')
fig.show()

# --- Plotly Chart 2 (NEW 3D): Interactive Efficient Frontier ---
print("\n--- Generating Plotly Chart 2 (3D): Interactive Efficient Frontier ---")
ef_model = MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, efficient_frontier_size=100)
ef_model.fit(X_train)
efficient_frontier = ef_model.predict(X_train)
ef_data = {
    'Annualized Volatility': [p.annualized_standard_deviation for p in efficient_frontier],
    'Annualized Mean Return': [p.annualized_mean for p in efficient_frontier],
    'Portfolio Type': 'Efficient Frontier'
}
efficient_frontier_df = pd.DataFrame(ef_data)

# Calculate the Tangency Portfolio for marking
tangency_portfolio_on_train = mean_risk_model.predict(X_train)

# Create a 3D scatter plot
fig = go.Figure(data=[
    go.Scatter3d(
        x=efficient_frontier_df['Annualized Volatility'],
        y=efficient_frontier_df['Annualized Mean Return'],
        z=[0] * len(efficient_frontier_df),  # Add a dummy Z-axis for 3D visualization
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=4)
    ),
    go.Scatter3d(
        x=[tangency_portfolio_on_train.annualized_standard_deviation],
        y=[tangency_portfolio_on_train.annualized_mean],
        z=[0],  # Dummy Z-axis
        mode='markers',
        name='Max Sharpe (Tangency) Portfolio',
        marker=dict(symbol='diamond', size=15, color='red')  # CHANGE 'star' to a valid symbol like 'diamond'
    )
])

# Configure the layout
fig.update_layout(
    title='Interactive 3D Efficient Frontier',
    scene=dict(
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Mean Return',
        zaxis_title='Dummy Axis',
        xaxis=dict(gridcolor='rgba(0,0,0,0.2)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.2)'),
        zaxis=dict(gridcolor='rgba(0,0,0,0.2)', showticklabels=False), # Hide tick labels for the dummy axis
        aspectmode='cube'
    ),
    height=700,
    margin=dict(l=0, r=0, b=0, t=50)
)
fig.show()

# --- Plotly Chart 3: Asset Weights (Grouped Bar Chart) ---
print("\n--- Generating Plotly Chart 3: Asset Weights ---")
weights_data = []
for i, asset in enumerate(X_train.columns):
    weights_data.append({'Portfolio Strategy': 'Mean-Risk (Full Assets)', 'Asset': asset, 'Weight': mean_risk_model.weights_[i]})
for i, asset in enumerate(X_train.columns):
    weights_data.append({'Portfolio Strategy': 'HRP (Full Assets)', 'Asset': asset, 'Weight': hrp_model.weights_[i]})
ew_full_weights = np.array([1 / len(X_train.columns)] * len(X_train.columns))
for i, asset in enumerate(X_train.columns):
    weights_data.append({'Portfolio Strategy': 'Equal Weighted (Full Assets)', 'Asset': asset, 'Weight': ew_full_weights[i]})
weights_df = pd.DataFrame(weights_data)
fig = px.bar(weights_df, x='Asset', y='Weight', color='Portfolio Strategy', barmode='group',
             title='Asset Weights by Portfolio Strategy')
fig.show()

# --- Plotly Chart 4: Risk Contributions (Bar Chart) ---
print("\n--- Generating Plotly Chart 4: Risk Contributions ---")
risk_contributions_data = []
# Mean-Risk Portfolio
weights_mr = mean_risk_model.weights_
covariance_mr = X_train.cov()
cov_weights_product_mr = np.dot(covariance_mr, weights_mr)
portfolio_risk_mr = np.dot(weights_mr.T, cov_weights_product_mr)
asset_contributions_mr = weights_mr * cov_weights_product_mr / portfolio_risk_mr
for i, asset in enumerate(X_train.columns):
    risk_contributions_data.append({
        'Portfolio Strategy': 'Mean-Risk (Full Assets)',
        'Asset': asset,
        'Risk Contribution': asset_contributions_mr[i]
    })
# HRP Portfolio
weights_hrp = hrp_model.weights_
covariance_hrp = X_train.cov()
cov_weights_product_hrp = np.dot(covariance_hrp, weights_hrp)
portfolio_risk_hrp = np.dot(weights_hrp.T, cov_weights_product_hrp)
asset_contributions_hrp = weights_hrp * cov_weights_product_hrp / portfolio_risk_hrp
for i, asset in enumerate(X_train.columns):
    risk_contributions_data.append({
        'Portfolio Strategy': 'HRP (Full Assets)',
        'Asset': asset,
        'Risk Contribution': asset_contributions_hrp[i]
    })
risk_contributions_df = pd.DataFrame(risk_contributions_data)
fig = px.bar(
    risk_contributions_df,
    x='Asset',
    y='Risk Contribution',
    color='Portfolio Strategy',
    barmode='group',
    title='Risk Contribution of Each Asset by Portfolio Strategy (Test Set)'
)
fig.show()

# --- Plotly Chart 5: Correlation Heatmap ---
print("\n--- Generating Plotly Chart 5: Correlation Heatmap ---")
correlation_matrix = X_train.corr()
fig = px.imshow(
    correlation_matrix,
    text_auto=True,
    title='Asset Correlation Matrix (Training Data)',
    color_continuous_scale='RdBu_r',
    aspect="auto",
    labels=dict(x="Asset", y="Asset", color="Correlation")
)
fig.update_layout(xaxis_nticks=len(X_train.columns), yaxis_nticks=len(X_train.columns))
fig.show()

# --- Plotly Chart 6: Distribution of Portfolio Returns (Histogram) ---
print("\n--- Generating Plotly Chart 6: Distribution of Returns ---")
portfolio_returns_df = pd.DataFrame({
    'Daily Returns': portfolio_mean_risk_test.returns,
    'Strategy': 'Mean-Risk (Full Assets)'
})
fig = px.histogram(
    portfolio_returns_df,
    x="Daily Returns",
    nbins=50,
    title='Distribution of Daily Returns for Mean-Risk Portfolio',
    marginal="box",
    color='Strategy',
    labels={"Daily Returns": "Daily Returns (%)"}
)
fig.show()

# --- Plotly Chart 7: Drawdown Analysis ---
print("\n--- Generating Plotly Chart 7: Drawdown Analysis ---")
def calculate_drawdowns(cumulative_returns):
    peak = np.maximum.accumulate(1 + cumulative_returns)
    drawdown = (1 + cumulative_returns - peak) / peak
    return drawdown

drawdown_data = []
drawdown_data.append(pd.DataFrame({
    'Date': X_test.index,
    'Drawdown': calculate_drawdowns(portfolio_mean_risk_test.cumulative_returns),
    'Portfolio Strategy': 'Mean-Risk (Full Assets)'
}))
drawdown_data.append(pd.DataFrame({
    'Date': X_test.index,
    'Drawdown': calculate_drawdowns(portfolio_hrp_test.cumulative_returns),
    'Portfolio Strategy': 'HRP (Full Assets)'
}))
drawdown_df = pd.concat(drawdown_data, ignore_index=True)
fig = px.line(
    drawdown_df,
    x='Date',
    y='Drawdown',
    color='Portfolio Strategy',
    title='Portfolio Drawdowns Over Time (Test Set)',
    labels={"Drawdown": "Drawdown (%)"},
    hover_data={'Drawdown': ':.2%'}
)
fig.show()

# --- Plotly Chart 8: Individual Asset Performance (Scatter Plot) ---
print("\n--- Generating Plotly Chart 8: Individual Asset Performance ---")
individual_asset_performance_data = []
for col in X_test.columns:
    asset_returns = X_test[col]
    asset_mean_return = asset_returns.mean() * 252
    asset_std_dev = asset_returns.std() * np.sqrt(252)
    asset_sharpe = (asset_returns.mean() - daily_risk_free_rate) / asset_returns.std() * np.sqrt(252) if asset_returns.std() > 0 else 0
    individual_asset_performance_data.append({
        'Asset': col,
        'Annualized Mean Return': asset_mean_return,
        'Annualized Volatility': asset_std_dev,
        'Annualized Sharpe Ratio': asset_sharpe
    })
individual_asset_perf_df = pd.DataFrame(individual_asset_performance_data)
fig = px.scatter(individual_asset_perf_df, x='Annualized Volatility', y='Annualized Mean Return',
                 text='Asset', size='Annualized Sharpe Ratio',
                 title='Individual Asset Performance (Return vs. Volatility)',
                 hover_data=['Annualized Sharpe Ratio'])
fig.update_traces(textposition='top center')
fig.show()

# --- Plotly Chart 9: Interactive Cumulative Returns Animation ---
print("\n--- Generating Plotly Chart 9: Interactive Cumulative Returns Animation ---")
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
cumulative_returns_combined_df['Date_str'] = cumulative_returns_combined_df['Date'].dt.strftime('%Y-%m-%d')
fig = px.line(
    cumulative_returns_combined_df,
    x="Date",
    y="Cumulative Return",
    color="Portfolio Strategy",
    animation_frame="Date_str",
    title="Cumulative Returns Performance over Time (Test Set)",
    labels={"Cumulative Return": "Cumulative Return (Normalized to 1)"},
    markers=True
)
fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return", legend_title="Portfolio Strategy", transition_duration=500)
fig.show()