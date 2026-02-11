#!/usr/bin/env python3
"""
Script for running grid search over multiple hyperparameter combinations.
"""

import argparse
import itertools
import gc
import psutil
from datetime import datetime
from run_model_suffix import run_model_suffix_pipeline
import grid_search_configs as configs


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
# Or use one of the predefined configurations from grid_search_configs.py
PARAM_GRID = {
    'target_activity': ['sphereActivity', 'ladderActivity'],
    'test_activity': ['sphereActivity', 'ladderActivity'],
    'threshold': [0.5, 0.6, 0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [5,6,7,8,9,10],
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2, 4, 6, 8, 10],
    'ghost_exp': [4],
    'pmin': [0.000001, 0.00001, 0.0001],
    'gamma_min': [0.0, 0.000001, 0.0001],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}


def generate_combinations(param_grid, limit=None):
    """
    Generate all possible combinations of parameters from the grid.
    
    Args:
        param_grid: Dictionary with parameter names as keys and lists of values
        limit: Maximum number of combinations to generate (None for all)
        
    Returns:
        List of dictionaries, each containing one parameter combination
    """
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for i, combination in enumerate(itertools.product(*values)):
        if limit is not None and i >= limit:
            break
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def run_grid_search(param_grid, dry_run=False, verbose=True, max_memory_mb=None, limit=None):
    """
    Run the model pipeline for all parameter combinations.
    
    Args:
        param_grid: Dictionary with parameter names as keys and lists of values
        dry_run: If True, only print combinations without running
        verbose: If True, print detailed progress information
        max_memory_mb: If set, warn when memory usage exceeds this threshold (in MB)
        limit: Maximum number of combinations to run (None for all)
    """
    combinations = generate_combinations(param_grid, limit=limit)
    total_combinations = len(combinations)
    
    print("=" * 80)
    print(f"GRID SEARCH CONFIGURATION")
    print("=" * 80)
    print(f"Total combinations to run: {total_combinations}\n")
    
    if limit:
        total_possible = 1
        for values in param_grid.values():
            total_possible *= len(values)
        if total_combinations < total_possible:
            print(f"Limited to first {total_combinations} of {total_possible} possible combinations\n")
    
    if verbose:
        print("Parameter grid:")
        for param, values in param_grid.items():
            if len(values) > 1:
                print(f"  {param}: {values}")
        print()
    
    if dry_run:
        print("DRY RUN - Combinations that would be executed:")
        for i, params in enumerate(combinations, 1):
            print(f"\n[{i}/{total_combinations}]")
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
        print("\n" + "=" * 80)
        print(f"RUNNING COMBINATION {i}/{total_combinations}")
        print("=" * 80)
        
        if verbose:
            print("Parameters:")
            for key, value in params.items():
                if param_grid[key] and len(param_grid[key]) > 1:
                    print(f"  {key}: {value}")
            print()
        
        try:
            # Run the pipeline with current parameter combination
            run_model_suffix_pipeline(**params)
            successful_runs += 1
            print(f"\n✓ Combination {i}/{total_combinations} completed successfully")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            failed_runs.append((i, params, str(e)))
            print(f"\n✗ Combination {i}/{total_combinations} failed with error:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print(f"\n  Full traceback:")
            print(error_traceback)
            print("  Continuing with next combination...")
        
        finally:
            # Clean up memory after each run (success or failure)
            print("\nCleaning up memory...")
            cleanup_memory(verbose=verbose)
            
            # Check memory usage
            current_memory = get_memory_usage()
            if max_memory_mb and current_memory > max_memory_mb:
                memory_warnings += 1
                print(f"WARNING: Memory usage ({current_memory:.1f} MB) exceeds threshold ({max_memory_mb} MB)")
                print(f"Consider reducing batch size or restarting the script")
            
            if verbose and i < total_combinations:
                print(f"  Ready for next combination ({i+1}/{total_combinations})\n")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    final_memory = get_memory_usage()
    
    print("\n" + "=" * 80)
    print("GRID SEARCH SUMMARY")
    print("=" * 80)
    print(f"Total combinations: {total_combinations}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Memory warnings: {memory_warnings}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ {final_memory - initial_memory:+.1f} MB)")
    
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
        description='Run grid search over hyperparameter combinations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show available predefined configurations
  python3 run_grid_search.py --list-configs
  
  # Dry run with default configuration
  python3 run_grid_search.py --dry-run
  
  # Run with a predefined configuration
  python3 run_grid_search.py --run --config GRID_K_VALUES
  
  # Run with default configuration
  python3 run_grid_search.py --run
  
  # Run with memory monitoring (warning at 8GB)
  python3 run_grid_search.py --run --max-memory 8192
  
  # Run only first 10 combinations (useful for testing)
  python3 run_grid_search.py --run --limit 10
  
  # Run with minimal output
  python3 run_grid_search.py --run --quiet
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
    
    args = parser.parse_args()
    
    # List available configurations
    if args.list_configs:
        print("Available predefined configurations:\n")
        config_names = [name for name in dir(configs) if name.startswith('GRID_')]
        for config_name in sorted(config_names):
            config = getattr(configs, config_name)
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
        if not hasattr(configs, args.config):
            print(f"Error: Configuration '{args.config}' not found.")
            print("Use --list-configs to see available configurations.")
            return
        param_grid = getattr(configs, args.config)
        print(f"Using predefined configuration: {args.config}\n")
    else:
        param_grid = PARAM_GRID
        print("Using default configuration from PARAM_GRID\n")
    
    verbose = not args.quiet
    
    run_grid_search(
        param_grid, 
        dry_run=args.dry_run, 
        verbose=verbose,
        max_memory_mb=args.max_memory,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
