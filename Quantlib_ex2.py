import QuantLib as ql
import numpy as np
import datetime

# --- 0. Set the Evaluation Date ---
today = ql.Date(2, 8, 2025) # Day, Month (number), Year
ql.Settings.instance().evaluationDate = today

# --- 1. Define Market Data (as used in previous examples to create the surface) ---
calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
day_count = ql.Actual365Fixed()

expiry_dates = [
    calendar.advance(today, ql.Period(3, ql.Months)),
    calendar.advance(today, ql.Period(6, ql.Months)),
    calendar.advance(today, ql.Period(1, ql.Years)),
    calendar.advance(today, ql.Period(2, ql.Years))
]

strikes = [90.0, 100.0, 110.0, 120.0]

vol_matrix_np = np.array([
    [0.25, 0.22, 0.21, 0.23],
    [0.23, 0.2,  0.19, 0.21],
    [0.22, 0.19, 0.18, 0.2 ],
    [0.21, 0.18, 0.17, 0.19]
])

rows, cols = vol_matrix_np.shape
ql_vol_matrix = ql.Matrix(rows, cols)
for i in range(rows):
    for j in range(cols):
        ql_vol_matrix[i][j] = vol_matrix_np[i, j]

# --- Create the BlackVarianceSurface (as per our last successful fix) ---
vol_surface = ql.BlackVarianceSurface(
    today,          # 1. referenceDate (ql.Date)
    calendar,       # 2. calendar (ql.Calendar)
    expiry_dates,   # 3. Vector of expiry dates (list of ql.Date) - MOVED HERE!
    strikes,        # 4. Vector of strike prices (list of float)
    ql_vol_matrix,  # 5. ql.Matrix
    day_count       # 6. dayCounter (ql.DayCounter) - MOVED HERE!
)
print("BlackVarianceSurface created successfully.")


# --- EXAMPLE OF USING THE PRICING ENGINE WITH BLACKVARIANCESURFACE ---

# Define the option (e.g., European Call)
option_type = ql.Option.Call
strike_option = 105.0 # Example strike
expiration_date_option = calendar.advance(today, ql.Period(1, ql.Years)) # Example 1-year expiry

payoff = ql.PlainVanillaPayoff(option_type, strike_option)
exercise = ql.EuropeanExercise(expiration_date_option)
european_option = ql.VanillaOption(payoff, exercise)

# Market data for the underlying process
spot_price = ql.QuoteHandle(ql.SimpleQuote(100.0))
risk_free_rate = 0.05
dividend_yield = 0.02

# Flat term structures for rates/dividends (simplified for this example)
flat_risk_free_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(today, risk_free_rate, day_count)
)
flat_dividend_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(today, dividend_yield, day_count)
)

# --- Crucially, use the constructed volatility surface here ---
# Wrap the BlackVarianceSurface in a handle to use it as a BlackVolTermStructure
vol_surface_handle = ql.BlackVolTermStructureHandle(vol_surface)

# Create the Black-Scholes-Merton process
# This process uses all the market data, including your volatility surface
bsm_process = ql.BlackScholesMertonProcess(
    spot_price,
    flat_dividend_ts,
    flat_risk_free_ts,
    vol_surface_handle # The BlackVolTermStructureHandle is passed here!
)

# Choose a pricing engine (AnalyticEuropeanEngine for European options)
pricing_engine = ql.AnalyticEuropeanEngine(bsm_process)

# Set the pricing engine for the option
european_option.setPricingEngine(pricing_engine)

# Get the option price and Greeks
option_price = european_option.NPV()
print(f"\nOption Price (Strike={strike_option}, Expiry={expiration_date_option.ISO()}) using Volatility Surface: {option_price:.4f}")

# Greeks (risk sensitivities)
print(f"  Delta: {european_option.delta():.4f}")
print(f"  Gamma: {european_option.gamma():.4f}")
print(f"  Vega: {european_option.vega():.4f}")
print(f"  Theta: {european_option.theta():.4f}")
print(f"  Rho: {european_option.rho():.4f}")