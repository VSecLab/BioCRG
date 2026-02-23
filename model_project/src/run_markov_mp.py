import os 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import sys
import psutil
import itertools
import pandas as pd

from contextlib import contextmanager
from multiprocessing import Process, Lock 
from run_model_markov_nofile import run_model_pipeline
from data_processing.src import config


N_PROCESSES = 6
BATCH_SIZE = 5
FILE_PATH = config.RESULTS_DIR + f"/grid_results/results_markov_mp.csv"

EXPECTED_COLUMNS = [
    "Username", "Features", "Target_Activity", "TestActivity", "Target_Threshold", 
    "Test_Threshold", "Clustering", "eps", "min_samples", "K", "scaler", 
    "LogNumber", "N_states", "Ghost_Count", "Ghost_Exp", "Concat", "AvgLogProb", "L"
]

PARAM_GRID = {
    'target_activity': ['sphereActivity', 'ladderActivity'],
    'test_activity': ['sphereActivity', 'ladderActivity'],
    'threshold': [0.5, 0.6, 0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [5, 6, 7, 8, 9, 10],
    'eps': [0.25],
    'min_samples': [11],
    'ghost_exp': [4],
    'enable_plots': [False],
    'save_output': [False]
}


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB


def cleanup_memory(verbose=False, process_id=None):
    """Force garbage collection and clear memory."""
    if verbose:
        mem_before = get_memory_usage()
    
    # Force garbage collection
    gc.collect()
    
    if verbose:
        mem_after = get_memory_usage()
        prefix = f"Process {process_id}: " if process_id is not None else ""
        print(f"{prefix}Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (freed {mem_before - mem_after:.1f} MB)")


def generate_combinations(param_grid, limit=None, skip=0):
    """
    Generate all possible combinations of parameters from the grid.
    
    Args:
        param_grid: Dictionary with parameter names as keys and lists of values
        limit: Maximum number of combinations to generate (None for all)
        skip: Number of initial combinations to skip (useful for resuming)
        
    Returns:
        List of dictionaries, each containing one parameter combination
    """
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for i, combination in enumerate(itertools.product(*values)):
        if i < skip:
            continue
        if limit is not None and len(combinations) >= limit:
            break
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def chunk_list(lst, n):
    """Split a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def initialize_output_file(filename, columns):
    """
    Initialize the output CSV file with headers.
    
    Args:
        filename: Path to the output file
        columns: List of column names
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Crea un DataFrame vuoto con le colonne specificate e scrivilo
    pd.DataFrame(columns=columns).to_csv(filename, index=False)
    print(f"Initialized output file: {filename}")


@contextmanager
def suppress_stdout():
    """Context manager to suppress all print statements."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def worker(process_id, param_chunk, batch_size, filename, lock):
    """
    Worker function for parallel processing.
    
    Args:
        process_id: ID of the worker process
        param_chunk: Chunk of parameter combinations to process
        batch_size: Number of combinations to process before writing to file
        filename: Path to the output file
        lock: Lock for synchronized file writing
    """
    i = 1
    for batch in chunk_list(param_chunk, batch_size):
        print(f"Process {process_id}: Starting batch {i} with {len(batch)} combinations...")
        # with suppress_stdout():
        try:
            results = []
             
            for params in batch:
                try:
                    result = run_model_pipeline(process_id=process_id, **params)
                    results.append(result)
                except Exception as e:
                    # print(f"Process {process_id}: ERROR in combination {params}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continua con le altre combinazioni invece di bloccare tutto
                    continue
            
            if not results:
                # print(f"Process {process_id}: No valid results in batch {i}, skipping write")
                continue
                
            dfs = [res['final_results'] for res in results]
            batch_df = pd.concat(dfs, ignore_index=True)

            # Validazione: assicurati che le colonne siano nell'ordine corretto
            if not all(col in batch_df.columns for col in EXPECTED_COLUMNS):
                missing = set(EXPECTED_COLUMNS) - set(batch_df.columns)
                print(f"Process {process_id}: WARNING - Missing columns: {missing}")
           
            batch_df = batch_df.reindex(columns=EXPECTED_COLUMNS)
            # print(f"Process {process_id}: Batch {i} - DataFrame shape: {batch_df.shape} - Columns: {batch_df.columns.tolist()}")
            
            with lock:
                batch_df.to_csv(filename, mode='a', header=False, index=False, na_rep='NaN')

            print(f"Process {process_id}: Completed batch {i} of {len(batch)} combinations. Rows written: {len(batch_df)}")
            
        except Exception as e:
            print(f"Process {process_id}: CRITICAL ERROR in batch {i}: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            cleanup_memory(verbose=False, process_id=process_id)
            i += 1
    
    print(f"Process {process_id}: Finished all batches.")


def main(): 
    """Main function to run parallel grid search with Markov Chain model."""
    lock = Lock()
    combinations = generate_combinations(PARAM_GRID)
    chunk_size = max(1, len(combinations) // N_PROCESSES)

    print(f"Total combinations: {len(combinations)}")
    print(f"Number of processes: {N_PROCESSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output file: {FILE_PATH}")
    print("=" * 60)

    initialize_output_file(FILE_PATH, EXPECTED_COLUMNS)

    process_chunks = [combinations[i*chunk_size:(i+1)*chunk_size] for i in range(N_PROCESSES)]
    # Aggiungi le rimanenti all'ultimo chunk
    if len(combinations) > N_PROCESSES * chunk_size:
        process_chunks[-1].extend(combinations[N_PROCESSES * chunk_size:])

    processes = []
    for i, chunk in enumerate(process_chunks):
        p = Process(target=worker, args=(i, chunk, BATCH_SIZE, FILE_PATH, lock))
        processes.append(p)
        p.start()
        print(f"Started Process {i} with {len(chunk)} combinations")

    for p in processes:
        p.join()

    print("\n" + "=" * 60)
    print("All processes completed!")
    for i, p in enumerate(processes):
        print(f"Process {i} exit code: {p.exitcode}")
    print("=" * 60)


if __name__ == "__main__":
    main()
