"""
Predefined grid search configurations for different experimental scenarios.

To use a configuration, import it and pass it to run_grid_search:
    from grid_search_configs import GRID_K_VALUES
    run_grid_search(GRID_K_VALUES)
"""

# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

# 1. Test different K values for KMeans
GRID_K_VALUES = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [5, 8, 10, 12, 15],  # Variable
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2],
    'ghost_exp': [4],
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}

# 2. Test different L values for suffix tree
GRID_L_VALUES = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [10],
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2, 3, 4, 5],  # Variable
    'ghost_exp': [4],
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}

# 3. Test different thresholds
GRID_THRESHOLDS = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.5, 0.6, 0.7, 0.8, 0.9],  # Variable
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [10],
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2],
    'ghost_exp': [4],
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}

# 4. Test different feature modes
GRID_FEATURES = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.7],
    'scaler': ['standard'],
    'feature_mode': ['rotation', 'position', 'both'],  # Variable
    'clustering_algorithm': ['kmeans'],
    'k_value': [10],
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2],
    'ghost_exp': [4],
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}

# 5. Test different clustering algorithms
GRID_CLUSTERING = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans', 'dbscan'],  # Variable
    'k_value': [10],
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2],
    'ghost_exp': [4],
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}

# 6. Test suffix tree parameters (pmin, gamma_min, eps_suffix)
GRID_SUFFIX_TREE_PARAMS = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [10],
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2],
    'ghost_exp': [4],
    'pmin': [0.000001, 0.00001, 0.0001],  # Variable
    'gamma_min': [0.0, 0.01, 0.05],  # Variable
    'eps_suffix': [0.0, 0.01, 0.05],  # Variable
    'enable_plots': [False]
}

# 7. Test ghost exponent values
GRID_GHOST_EXP = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [10],
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2],
    'ghost_exp': [2, 3, 4, 5, 6],  # Variable
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}

# 8. Comprehensive search (K and L values)
GRID_K_AND_L = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [5, 8, 10, 12, 15],  # Variable
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2, 3, 4],  # Variable
    'ghost_exp': [4],
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}

# 9. Test different activities
GRID_ACTIVITIES = {
    'target_activity': ['sphereActivity', 'ladderActivity', 'trashActivity'],  # Variable
    'test_activity': ['sphereActivity', 'ladderActivity', 'trashActivity'],  # Variable
    'threshold': [0.7],
    'scaler': ['standard'],
    'feature_mode': ['both'],
    'clustering_algorithm': ['kmeans'],
    'k_value': [10],
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2],
    'ghost_exp': [4],
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}

# 10. Full comprehensive search (WARNING: This will take a lot of time!)
GRID_COMPREHENSIVE = {
    'target_activity': ['sphereActivity'],
    'test_activity': ['sphereActivity'],
    'threshold': [0.6, 0.7, 0.8],  # Variable
    'scaler': ['standard'],
    'feature_mode': ['rotation', 'position', 'both'],  # Variable
    'clustering_algorithm': ['kmeans'],
    'k_value': [8, 10, 12],  # Variable
    'eps': [0.25],
    'min_samples': [11],
    'l_value': [2, 3, 4],  # Variable
    'ghost_exp': [3, 4, 5],  # Variable
    'pmin': [0.000001],
    'gamma_min': [0.0],
    'eps_suffix': [0.0],
    'enable_plots': [False]
}
