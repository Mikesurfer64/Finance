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
yf_data = None # Initialize yf_data to None

max_retries = 3
retry_delay_seconds = 60

for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}/{max_retries} to download data from Yahoo Finance...")
        yf_data = yf.download(yf_tickers, start=yf_start_date, end=yf_end_date)
        if not yf_data.empty:
            print("Successfully downloaded Yahoo Finance data.")
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
    if attempt == max_retries - 1 and yf_data is None:
        print("Failed to download Yahoo Finance data after multiple retries. Proceeding without this data.")

if yf_data is not None and not yf_data.empty:
    yf_data.to_csv("yahoo_finance_data.csv")
    print("Yahoo Finance data saved to yahoo_finance_data.csv")
else:
    print("No Yahoo Finance data to save or proceed with.")


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
plt.show()

q = 0.5
budget = num_assets // 2
penalty = num_assets

portfolio = PortfolioOptimization(
    expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
)
qp = portfolio.to_quadratic_program()
print("\nQuadratic Program (QP) created:")
print(qp)


# --- CORRECTED print_result function ---
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
        # Adjusting for potential differences in eigenstate representation in older Qiskit versions
        # For NumPyMinimumEigensolver, eigenstate is typically a Statevector or an array/dict
        # For SamplingVQE/QAOA with Sampler, eigenstate might be a QuasiDistribution directly
        probabilities = {}
        if isinstance(eigenstate, QuasiDistribution):
            probabilities = eigenstate.binary_probabilities()
        elif hasattr(eigenstate, 'to_dict'): # For Statevector or similar objects that can convert to dict
            probabilities = {k: np.abs(v)**2 for k, v in eigenstate.to_dict().items()}
        elif isinstance(eigenstate, np.ndarray) and eigenstate.ndim == 1: # For simple array representations
            # If eigenstate is a numpy array of amplitudes, square them
            probabilities = {bin(i)[2:].zfill(len(eigenstate).bit_length()-1): np.abs(val)**2
                             for i, val in enumerate(eigenstate) if np.abs(val)**2 > 1e-9} # Only significant probs
            # This conversion to bitstring keys needs the number of qubits to zfill correctly
            # It's safer to rely on Qiskit's built-in methods if available.
        else:
            print("DEBUG: Eigenstate type not directly supported for probability extraction.")
            print(f"DEBUG: Eigenstate type: {type(eigenstate)}")


        if probabilities: # Ensure probabilities dict is not empty before printing
            print("\n----------------- Full result ---------------------")
            print("selection\tvalue\t\tprobability")
            print("---------------------------------------------------")
            probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

            for k, v in probabilities:
                x = np.array([int(i) for i in list(reversed(k))])
                # Ensure portfolio.to_quadratic_program() is accessible if 'portfolio' is global
                # Or pass qp as an argument to print_result
                try:
                    value = portfolio.to_quadratic_program().objective.evaluate(x)
                    print("%10s\t%.4f\t\t%.4f" % (x, value, v))
                except Exception as e:
                    print(f"Error evaluating objective for {x}: {e}")
                    print("%10s\t(N/A)\t\t%.4f" % (x, v))
        else:
            print("No probabilities to display from eigenstate.")
    else:
        print("\nSkipping probability printing as eigenstate was not available or could not be processed.")


# --- Classical Solver ---
print("\n--- Running Classical Solver (NumPyMinimumEigensolver) ---")
exact_mes = NumPyMinimumEigensolver()
exact_eigensolver = MinimumEigenOptimizer(exact_mes)
result = exact_eigensolver.solve(qp)
print_result(result)

# --- Quantum Solver: SamplingVQE ---
print("\n--- Running Quantum Solver (SamplingVQE) ---")
algorithm_globals.random_seed = 1234
cobyla = COBYLA()
cobyla.set_options(maxiter=500)
ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
svqe = MinimumEigenOptimizer(svqe_mes)
result = svqe.solve(qp)
print_result(result)

# --- Quantum Solver: QAOA ---
print("\n--- Running Quantum Solver (QAOA) ---")
algorithm_globals.random_seed = 1234
cobyla = COBYLA()
cobyla.set_options(maxiter=250)
qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)
print_result(result)