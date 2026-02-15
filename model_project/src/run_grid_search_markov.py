#!/usr/bin/env python3
"""
Script for running grid search over multiple hyperparameter combinations using Markov Chain model.
"""

import argparse
import itertools
import gc
import psutil
from datetime import datetime
from run_model import run_model_pipeline


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB


def cleanup_memory(verbose=True):
    """Force garbage collection and clear memory."""
    if verbose:
        mem_before = get_memory_usage()
    
    # Force garbage collection
    gc.collect()
    
    if verbose:
        mem_after = get_memory_usage()
        print(f"  Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (freed {mem_before - mem_after:.1f} MB)")


# ============================================================================
# GRID SEARCH CONFIGURATION
# ============================================================================

# Define parameter grids for each hyperparameter
# Add/remove values in the lists to configure your grid search


GRID_FULL = {
    'target_activity': ['sphereActivity', 'ladderActivity'],
    'test_activity': ['sphereActivity', 'ladderActivity'],
    'threshold': [0.5, 0.6, 0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [5,6,7,8,9,10],
    'eps': [0.25],
    'min_samples': [11],
    'ghost_exp': [4],
    'enable_plots': [False]
}
# Default parameter grid
PARAM_GRID = GRID_FULL


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


def run_grid_search(param_grid, dry_run=False, verbose=True, max_memory_mb=None, limit=None, skip=0):
    """
    Run the model pipeline for all parameter combinations.
    
    Args:
        param_grid: Dictionary with parameter names as keys and lists of values
        dry_run: If True, only print combinations without running
        verbose: If True, print detailed progress information
        max_memory_mb: If set, warn when memory usage exceeds this threshold (in MB)
        limit: Maximum number of combinations to run (None for all)
        skip: Number of initial combinations to skip (useful for resuming)
    """
    combinations = generate_combinations(param_grid, limit=limit, skip=skip)
    total_combinations = len(combinations)
    
    # Calculate total possible combinations
    total_possible = 1
    for values in param_grid.values():
        total_possible *= len(values)
    
    print("=" * 80)
    print(f"GRID SEARCH CONFIGURATION (MARKOV CHAIN MODEL)")
    print("=" * 80)
    print(f"Total possible combinations: {total_possible}")
    if skip > 0:
        print(f"Skipping first {skip} combinations (resuming from {skip + 1})")
    print(f"Combinations to run: {total_combinations}\n")
    
    if limit and total_combinations < (total_possible - skip):
        print(f"Limited to {total_combinations} combinations\n")
    
    if verbose:
        print("Parameter grid:")
        for param, values in param_grid.items():
            if len(values) > 1:
                print(f"  {param}: {values}")
        print()
    
    if dry_run:
        print("DRY RUN - Combinations that would be executed:")
        for i, params in enumerate(combinations, 1):
            combination_number = skip + i
            print(f"\n[{combination_number}/{total_possible}] (run {i}/{total_combinations})")
            for key, value in params.items():
                if param_grid[key] and len(param_grid[key]) > 1:
                    print(f"  {key}: {value}")
        print("\nDry run completed. Use --run to execute.")
        return
    
    # Track results
    successful_runs = 0
    failed_runs = []
    memory_warnings = 0
    
    # Get initial memory
    initial_memory = get_memory_usage()
    if verbose:
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        if max_memory_mb:
            print(f"Memory warning threshold: {max_memory_mb} MB")
        print()
    
    start_time = datetime.now()
    print(f"Starting grid search at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for i, params in enumerate(combinations, 1):
        combination_number = skip + i
        print("\n" + "=" * 80)
        print(f"RUNNING COMBINATION {combination_number}/{total_possible} (run {i}/{total_combinations})")
        print("=" * 80)
        
        if verbose:
            print("Parameters:")
            for key, value in params.items():
                if param_grid[key] and len(param_grid[key]) > 1:
                    print(f"  {key}: {value}")
            print()
        
        try:
            # Run the pipeline with current parameter combination
            run_model_pipeline(**params)
            successful_runs += 1
            print(f"\nCombination {combination_number}/{total_possible} completed successfully")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            failed_runs.append((combination_number, params, str(e)))
            print(f"\n✗ Combination {combination_number}/{total_possible} failed with error:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print(f"\n  Full traceback:")
            print(error_traceback)
            print("  Continuing with next combination...")
        
        finally:
            # Clean up memory after each run (success or failure)
            c_mem = get_memory_usage()
            print("\nCleaning up memory...")
            
            cleanup_memory(verbose=verbose)
            
            # Check memory usage
            current_memory = get_memory_usage()

            print(f"Before cleanup memory usage: {c_mem:.1f} MB")
            print(f"After cleanup memory usage: {current_memory:.1f} MB")
            if max_memory_mb and current_memory > max_memory_mb:
                memory_warnings += 1
                print(f"WARNING: Memory usage ({current_memory:.1f} MB) exceeds threshold ({max_memory_mb} MB)")
                print(f"Consider reducing batch size or restarting the script")
            
            if verbose and i < total_combinations:
                print(f"  Ready for next combination ({combination_number+1}/{total_possible})\n")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    final_memory = get_memory_usage()
    
    print("\n" + "=" * 80)
    print("GRID SEARCH SUMMARY")
    print("=" * 80)
    print(f"Total possible combinations: {total_possible}")
    if skip > 0:
        print(f"Skipped: {skip}")
    print(f"Combinations run: {total_combinations}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Memory warnings: {memory_warnings}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ {final_memory - initial_memory:+.1f} MB)")
    if skip + total_combinations < total_possible:
        remaining = total_possible - skip - total_combinations
        print(f"\nTo continue from here, use: --skip {skip + total_combinations}")
        print(f"Remaining combinations: {remaining}")
    
    if failed_runs:
        print("\nFailed combinations:")
        for i, params, error in failed_runs:
            print(f"\n  Combination {i}:")
            for key, value in params.items():
                if param_grid[key] and len(param_grid[key]) > 1:
                    print(f"    {key}: {value}")
            print(f"    Error: {error}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run grid search over hyperparameter combinations (Markov Chain model)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show available predefined configurations
  python3 run_grid_search_markov.py --list-configs
  
  # Dry run with default configuration
  python3 run_grid_search_markov.py --dry-run
  
  # Run with a predefined configuration
  python3 run_grid_search_markov.py --run --config GRID_K_VALUES
  
  # Run with default configuration
  python3 run_grid_search_markov.py --run
  
  # Run with memory monitoring (warning at 8GB)
  python3 run_grid_search_markov.py --run --max-memory 8192
  
  # Run only first 10 combinations (useful for testing)
  python3 run_grid_search_markov.py --run --limit 10
  
  # Run with minimal output
  python3 run_grid_search_markov.py --run --quiet
        """
    )
    
    parser.add_argument('--list-configs', action='store_true',
                       help='List all available predefined configurations')
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--dry-run', action='store_true', 
                      help='Show all combinations without running')
    group.add_argument('--run', action='store_true',
                      help='Execute grid search')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Use a predefined configuration (e.g., GRID_K_VALUES)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (only summary)')
    
    parser.add_argument('--max-memory', type=int, default=None,
                       help='Warning threshold for memory usage in MB (e.g., 8192 for 8GB)')
    
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit the number of combinations to run (useful for testing)')
    
    parser.add_argument('--skip', type=int, default=0,
                       help='Skip the first N combinations (useful for resuming interrupted grid searches)')
    
    args = parser.parse_args()
    
    # List available configurations
    if args.list_configs:
        print("Available predefined configurations:\n")
        config_dict = {
            'GRID_FULL': GRID_FULL,
        }
        
        for config_name, config in sorted(config_dict.items()):
            # Count varying parameters
            varying_params = [k for k, v in config.items() if len(v) > 1]
            total_combinations = 1
            for v in config.values():
                total_combinations *= len(v)
            
            print(f"  {config_name}")
            print(f"    Combinations: {total_combinations}")
            print(f"    Varying parameters: {', '.join(varying_params)}")
            if varying_params:
                for param in varying_params:
                    print(f"      - {param}: {config[param]}")
            print()
        return
    
    # Require either --dry-run or --run if not listing configs
    if not args.dry_run and not args.run:
        parser.error("One of --dry-run, --run, or --list-configs is required")
    
    # Select configuration
    if args.config:
        config_dict = {
            'GRID_FULL': GRID_FULL,
        }
        
        if args.config not in config_dict:
            print(f"Error: Configuration '{args.config}' not found.")
            print("Use --list-configs to see available configurations.")
            return
        param_grid = config_dict[args.config]
        print(f"Using predefined configuration: {args.config}\n")
    else:
        param_grid = PARAM_GRID
        print("Using default configuration (GRID_FULL)\n")
    
    verbose = not args.quiet
    
    run_grid_search(
        param_grid, 
        dry_run=args.dry_run, 
        verbose=verbose,
        max_memory_mb=args.max_memory,
        limit=args.limit,
        skip=args.skip
    )


if __name__ == "__main__":
    main()
