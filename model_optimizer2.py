import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import pandas as pd
import time
import os # Import os for file path operations

# Qiskit imports (keep these as previously established for compatibility)
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider # Or YahooDataProvider if desired
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals # For random_seed

# --- Data Fetching Section (as previously corrected) ---

# Define tickers and date range for yfinance
yf_tickers = ["INTC", "AA", "XOM", "LVS"]
yf_start_date = "2023-01-01"
yf_end_date = "2024-01-01"
yf_data = None  # Initialize yf_data to None

# --- NEW LOGIC: Try to load from a local file first ---
data_file_path = "yahoo_finance_data.csv"
if os.path.exists(data_file_path):
    try:
        print(f"Local data file '{data_file_path}' found. Loading data from disk...")
        # Load the CSV, ensuring 'Date' column is parsed as a datetime index
        yf_data = pd.read_csv(data_file_path, index_col='Date', parse_dates=True)
        print("Successfully loaded data from local file.")
    except Exception as e:
        print(f"Error loading local data file: {e}. Attempting to download instead.")
        yf_data = None

# --- If local file load failed, proceed with download attempts ---
if yf_data is None or yf_data.empty:
    max_retries = 3
    retry_delay_seconds = 60
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to download data from Yahoo Finance...")
            yf_data = yf.download(yf_tickers, start=yf_start_date, end=yf_end_date)
            if not yf_data.empty:
                print("Successfully downloaded Yahoo Finance data.")
                yf_data.to_csv(data_file_path)
                print("Yahoo Finance data saved to local file for future use.")
                break
            else:
                print(f"Attempt {attempt + 1} downloaded empty data. Retrying...")
                time.sleep(retry_delay_seconds * (attempt + 1))
        except Exception as e:
            print(f"Failed to download Yahoo Finance data (Attempt {attempt + 1}): {e}")
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                print(f"Rate limited. Waiting for {retry_delay_seconds * (attempt + 1)} seconds before retrying...")
                time.sleep(retry_delay_seconds * (attempt + 1))
            else:
                print(f"Non-rate-limit error. Retrying in 10 seconds...")
                time.sleep(10)
    
    if yf_data is None or yf_data.empty:
        print("Failed to download Yahoo Finance data after all retries. Proceeding without this data.")
else:
    print("No Yahoo Finance data to save as it was loaded from a local file.")
    
# --- The rest of your code remains the same as before ---
# --- Remaining sections of your code... ---

# --- Qiskit Finance Portfolio Optimization Section ---

qiskit_cache_path = os.path.expanduser("~/.qiskit/qiskit_finance_cache.sqlite")
if os.path.exists(qiskit_cache_path):
    try:
        os.remove(qiskit_cache_path)
        print(f"Removed old Qiskit Finance cache: {qiskit_cache_path}")
    except OSError as e:
        print(f"Warning: Could not remove Qiskit Finance cache at {qiskit_cache_path}: {e}")
        print("This might lead to 'database is locked' errors if not resolved.")

num_assets = 4
seed = 123

stocks_for_random_data = [f"yf_tickers{i}" for i in range(num_assets)]
random_data_provider = RandomDataProvider(
    tickers=stocks_for_random_data,
    start=datetime.datetime(2023, 1, 1),
    end=datetime.datetime(2024, 1, 1),
    seed=seed,
)

print("Running RandomDataProvider...")
random_data_provider.run()
print("RandomDataProvider run complete.")

mu = random_data_provider.get_period_return_mean_vector()
sigma = random_data_provider.get_period_return_covariance_matrix()

plt.imshow(sigma, interpolation="nearest")
plt.title("Covariance Matrix")
plt.savefig('covariance_matrix.png')
plt.close()

q = 0.5
budget = num_assets // 2
penalty = num_assets

portfolio = PortfolioOptimization(
    expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
)
qp = portfolio.to_quadratic_program()
print("\nQuadratic Program (QP) created:")
print(qp)


# --- CORRECTED print_result function to return selection, probability, and value ---
def print_result(result):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    # Initialize eigenstate to None to prevent NameError
    eigenstate = None

    # Use a try-except block to safely access min_eigen_solver_result and eigenstate
    try:
        if hasattr(result, 'min_eigen_solver_result') and result.min_eigen_solver_result is not None:
            if hasattr(result.min_eigen_solver_result, 'eigenstate'):
                eigenstate = result.min_eigen_solver_result.eigenstate
            else:
                print("DEBUG: result.min_eigen_solver_result has no 'eigenstate' attribute.")
        else:
            print("DEBUG: result has no 'min_eigen_solver_result' attribute or it is None.")
    except Exception as e:
        # Catch any unexpected error during access
        print(f"DEBUG: An error occurred trying to access eigenstate: {e}")

    # Only proceed with probability printing if eigenstate was successfully obtained
    if eigenstate is not None:
        probabilities = {}
        if isinstance(eigenstate, QuasiDistribution):
            probabilities = eigenstate.binary_probabilities()
        elif hasattr(eigenstate, 'to_dict'): # For Statevector or similar objects that can convert to dict
            probabilities = {k: np.abs(v)**2 for k, v in eigenstate.to_dict().items()}
        elif isinstance(eigenstate, np.ndarray) and eigenstate.ndim == 1:
            probabilities = {bin(i)[2:].zfill(len(eigenstate).bit_length()-1): np.abs(val)**2
                             for i, val in enumerate(eigenstate) if np.abs(val)**2 > 1e-9}
        else:
            print("DEBUG: Eigenstate type not directly supported for probability extraction.")
            print(f"DEBUG: Eigenstate type: {type(eigenstate)}")


        if probabilities:
            print("\n----------------- Full result ---------------------")
            print("selection\tvalue\t\tprobability")
            print("---------------------------------------------------")
            probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            # Return the selection, probability, and value of the most probable state
            if probabilities:
                first_item_selection = np.array([int(i) for i in list(reversed(probabilities[0][0]))])
                first_item_probability = probabilities[0][1]
                
                for k, v in probabilities:
                    x = np.array([int(i) for i in list(reversed(k))])
                    try:
                        value_at_state = portfolio.to_quadratic_program().objective.evaluate(x)
                        print("%10s\t%.4f\t\t%.4f" % (x, value_at_state, v))
                    except Exception as e:
                        print(f"Error evaluating objective for {x}: {e}")
                        print("%10s\t(N/A)\t\t%.4f" % (x, v))
                return first_item_selection, first_item_probability, value

        else:
            print("No probabilities to display from eigenstate.")
            return selection, None, value
    else:
        print("\nSkipping probability printing as eigenstate was not available or could not be processed.")
        return selection, None, value


# --- Classical Solver ---
print("\n--- Running Classical Solver (NumPyMinimumEigensolver) ---")
exact_mes = NumPyMinimumEigensolver()
exact_eigensolver = MinimumEigenOptimizer(exact_mes)
result_exact = exact_eigensolver.solve(qp)
selection_exact, prob_exact, value_exact = print_result(result_exact)


# --- Quantum Solver: SamplingVQE ---
print("\n--- Running Quantum Solver (SamplingVQE) ---")
algorithm_globals.random_seed = 1234
cobyla = COBYLA()
cobyla.set_options(maxiter=500)
ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
svqe = MinimumEigenOptimizer(svqe_mes)
result_svqe = svqe.solve(qp)
selection_svqe, prob_svqe, value_svqe = print_result(result_svqe)


# --- Quantum Solver: QAOA ---
print("\n--- Running Quantum Solver (QAOA) ---")
algorithm_globals.random_seed = 1234
cobyla = COBYLA()
cobyla.set_options(maxiter=250)
qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result_qaoa = qaoa.solve(qp)
selection_qaoa, prob_qaoa, value_qaoa = print_result(result_qaoa)

# --- Plotting the minimum state probabilities ---
solvers = ['NumPyMinimumEigensolver', 'SamplingVQE', 'QAOA']
selections = [selection_exact, selection_svqe, selection_qaoa]
probabilities = [prob_exact, prob_svqe, prob_qaoa]

# Remove None values in case a solver failed
solvers_clean = [s for s, p in zip(solvers, probabilities) if p is not None]
selections_clean = [s for s, p in zip(selections, probabilities) if p is not None]
probabilities_clean = [p for p in probabilities if p is not None]

# Create the probability plot
plt.figure(figsize=(8, 6))
plt.bar(solvers_clean, probabilities_clean, color=['blue', 'green', 'red'])
plt.title('Probability of Found Minimum State for Each Solver')
plt.xlabel('Solver')
plt.ylabel('Probability')
plt.ylim(0, 1)

# Add labels to the bars
for i, prob in enumerate(probabilities_clean):
    selection_str = ' '.join(str(int(s)) for s in selections_clean[i])
    plt.text(i, prob, f'{prob:.4f}\n[{selection_str}]', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('minimum_state_probabilities.png')
plt.close()

# --- Graph Comparing the Objective Values ---

values = [value_exact, value_svqe, value_qaoa]

# Clean the data to remove any None values
solvers_clean_values = [s for s, v in zip(solvers, values) if v is not None]
values_clean = [v for v in values if v is not None]

# Create the new plot
plt.figure(figsize=(8, 6))
plt.bar(solvers_clean_values, values_clean, color=['blue', 'green', 'red'])
plt.title('Comparison of Objective Function Values')
plt.xlabel('Solver')
plt.ylabel('Objective Function Value (Lower is Better)')

# Add value labels to the bars
for i, value in enumerate(values_clean):
    plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('objective_value_comparison.png')
plt.close()

print("\nGraphs saved as 'stock_performance.png', 'minimum_state_probabilities.png', and 'objective_value_comparison.png'")