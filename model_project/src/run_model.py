#!/usr/bin/env python3
"""
Script for running the complete model pipeline using Markov Chain.
Based on model.ipynb workflow.
"""

import os
import sys
import argparse
import numpy as np 
import pandas as pd 
import kmeans as km
import transition as tr
import segmentation as sg
import validation_model as vm
import dbscan_clustering as db
import adjectives_handle as ah
import data_processing.src.plot as plot

from data_processing.src import config


def setup_features_and_paths(feature_mode):
    """
    Setup features and paths based on feature mode.
    
    Args:
        feature_mode: One of "rotation", "position", or "both"
        
    Returns:
        tuple: (features, segmentation_result, histogram_dir, feature_len)
    """
    if feature_mode == "rotation":
        features = config.ROTATION_FEATURES
        segmentation_result = config.ROTATION_SEGMENTATION_DIR
        histogram_dir = config.ROTATION_HISTOGRAMS_DIR
    elif feature_mode == "position":
        features = config.POSITION_FEATURES
        segmentation_result = config.POSITION_SEGMENTATION_DIR
        histogram_dir = config.POSITION_HISTOGRAMS_DIR
    else:
        features = config.FEATURES
        segmentation_result = config.BOTH_SEGMENTATION_DIR
        histogram_dir = config.BOTH_HISTOGRAMS_DIR
    
    feature_len = len(features) - 2
    return features, segmentation_result, histogram_dir, feature_len


def perform_segmentation(target_activity, features, threshold, scaler, segmentation_result, histogram_dir, enable_plots):
    """
    Perform segmentation and optionally plot histograms.
    
    Returns:
        tuple: (rsv_df, lsv_all_user, rsv_path)
    """
    rsv_df, lsv_all_user = sg.get_or_compute_segmentation(
        activity=target_activity,
        features=features,
        threshold=threshold,
        scaler=scaler,
        filepath=segmentation_result
    )
    
    print(rsv_df.head())
    
    os.makedirs(segmentation_result + f"/segments_number/{target_activity}", exist_ok=True)
    os.makedirs(histogram_dir + f"/{target_activity}", exist_ok=True)
    
    if enable_plots:
        plot.plot_histogram_of_segments(
            filepath=segmentation_result + f"/segments_number/{target_activity}/{scaler}/segmentation_results_{target_activity}_{threshold}_{scaler}.csv",
            savepath=histogram_dir + f"/{target_activity}"
        )
    
    rsv_path = segmentation_result + f"/right_singular_vector/{target_activity}/{scaler}/rsv_{target_activity}_{threshold}_{scaler}.csv"
    
    return rsv_df, lsv_all_user, rsv_path


def prepare_clustering_data(rsv_path):
    """
    Load and prepare data for clustering.
    
    Returns:
        tuple: (df, df_origin)
    """
    df_origin = pd.read_csv(rsv_path, index_col=None)
    df = df_origin.drop(columns=['Username', "Adjective", 'LogNumber'])
    df = df.dropna()
    df = df.drop_duplicates()
    
    return df, df_origin


def perform_clustering(df, df_origin, clustering_algorithm, k_value, eps, min_samples, 
                      target_activity, threshold, scaler, processed_data_path, enable_plots, feature_len):
    """
    Perform clustering (DBSCAN or KMeans) on the data.
    
    Returns:
        tuple: (df_origin with labels, df_no_outliers, unique_labels, result_directory)
    """
    print("=================================================\n")
    
    if enable_plots:
        if clustering_algorithm == "dbscan": 
            print("== DBSCAN Clustering Results ==\n")
            n_cols = len(df.columns)
            length = len(df)
            print(f"Number of columns: {n_cols}\nLength of DataFrame: {length}\n")
            db.grid(df)
        elif clustering_algorithm == "kmeans":
            print("== KMEANS Clustering Results ==\n")
            n_cols = len(df.columns)
            length = len(df)
            print(f"Number of columns: {n_cols}\nLength of DataFrame: {length}\n")
            km.grid(df)
    
    if clustering_algorithm == "dbscan":
        best_labels = db.dbscan_clustering(df, eps=eps, min_samples=min_samples)
        
        # Debug: verify dimensions
        print(f"DEBUG: df shape: {df.shape}, best_labels length: {len(best_labels)}")
        if len(best_labels) != len(df):
            raise ValueError(f"Labels length ({len(best_labels)}) doesn't match df length ({len(df)})")
        
        df['Label'] = best_labels
        
        feature_cols = df_origin.columns[-feature_len:].to_list()
        print(f"DEBUG: feature_cols (last {feature_len}): {feature_cols}")
        df_origin = df_origin.merge(df, on=feature_cols, how="left")
        
        # Check for NaN labels after merge
        if df_origin['Label'].isna().any():
            nan_count = df_origin['Label'].isna().sum()
            print(f"WARNING: {nan_count} rows have NaN labels after merge")
            print(f"df_origin rows: {len(df_origin)}, df rows: {len(df)}")
            # Fill NaN with -1 (outlier label)
            df_origin['Label'] = df_origin['Label'].fillna(-1)
        
        result_directory = "dbscan_results"
        dbscan_dir = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}/all_points/"
        os.makedirs(dbscan_dir, exist_ok=True)
        save_dir = dbscan_dir + f"dbscan_{target_activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv"
        
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        df_origin.to_csv(save_dir, index=False)
        
    elif clustering_algorithm == "kmeans":
        best_labels = km.kmeans_with_outlier_detection(df=df, K=k_value)
        
        # Debug: verify dimensions
        print(f"DEBUG: df shape: {df.shape}, best_labels length: {len(best_labels)}")
        if len(best_labels) != len(df):
            raise ValueError(f"Labels length ({len(best_labels)}) doesn't match df length ({len(df)})")
        
        df['Label'] = best_labels
        
        feature_cols = df_origin.columns[-feature_len:].to_list()
        print(f"DEBUG: feature_cols (last {feature_len}): {feature_cols}")
        df_origin = df_origin.merge(df, on=feature_cols, how="left")
        
        # Check for NaN labels after merge
        if df_origin['Label'].isna().any():
            nan_count = df_origin['Label'].isna().sum()
            print(f"WARNING: {nan_count} rows have NaN labels after merge")
            print(f"df_origin rows: {len(df_origin)}, df rows: {len(df)}")
            # Fill NaN with -1 (outlier label)
            df_origin['Label'] = df_origin['Label'].fillna(-1)
        
        result_directory = "kmeans_results"
        kmeans_dir = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}/all_points/"
        os.makedirs(kmeans_dir, exist_ok=True)
        save_dir = kmeans_dir + f"kmeans_{target_activity}_{threshold}_{scaler}_k{k_value}.csv"
        
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        df_origin.to_csv(save_dir, index=False)
    
    df_no_outliers = df.drop(df[df['Label'] == -1].index)
    unique_labels = df_no_outliers['Label'].unique()
    
    print(f"Number of unique labels (excluding outliers): {len(unique_labels)}\n")
    
    return df_origin, df_no_outliers, unique_labels, result_directory


def create_substantives_and_adjectives(df_no_outliers, unique_labels, df_origin, lsv_all_user, 
                                       best_labels, result_directory, target_activity, threshold, 
                                       scaler, processed_data_path, clustering_algorithm, eps, 
                                       min_samples, k_value):
    """
    Create substantives and adjectives dataframes.
    
    Returns:
        tuple: (df_substantives, df_adjectives, adj_dir)
    """
    df_substantives = db.get_substantives_dataframe(df=df_no_outliers, unique_labels=unique_labels)
    print("\n== Substantives DataFrame ==\n")
    print(df_substantives)
    
    if clustering_algorithm == "dbscan":
        save_dir = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}/substantives/substantives_{target_activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv"
    elif clustering_algorithm == "kmeans":
        save_dir = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}/substantives/substantives_{target_activity}_{threshold}_{scaler}_k{k_value}.csv"
    
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    df_substantives.to_csv(save_dir, index=False)
    
    df_new = df_origin.drop(columns=['Adjective']).drop_duplicates()
    lsv_clusters_dict = ah.adjectives_dict(best_labels, lsv_all_user, df_new)
    clusters_dict = ah.adjectives_segmentation(lsv_clusters_dict)
    df_adjectives = ah.generalize_adjectives(clusters=clusters_dict)
    
    print("\n== Adjectives DataFrame ==\n")
    print(df_adjectives)
    
    save_dir = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}/adjectives/"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    if clustering_algorithm == "dbscan":
        adj_dir = save_dir + f"adjectives_{target_activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv"
    elif clustering_algorithm == "kmeans":
        adj_dir = save_dir + f"adjectives_{target_activity}_{threshold}_{scaler}_k{k_value}.csv"
    
    df_adjectives.to_csv(adj_dir, index=False)
    
    return df_substantives, df_adjectives, adj_dir


def compute_transitions(clustering_algorithm, processed_data_path, result_directory, 
                       target_activity, scaler, threshold, k_value, eps, min_samples, enable_plots):
    """
    Compute transition matrix and probabilities for Markov Chain.
    
    Returns:
        tuple: (df_transition, df_probs, df_initial, sequence_stats)
    """
    file_path = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}"
    
    if clustering_algorithm == "dbscan":
        df_transition, df_probs, df_initial, sequence_stats = tr.transition_computation_dbscan(
            activity=target_activity,
            scaler=scaler,
            threshold=threshold,
            eps=eps,
            min_samples=min_samples,
            file_path=file_path,
            enable_plots=enable_plots
        )
    elif clustering_algorithm == "kmeans":
        df_transition, df_probs, df_initial, sequence_stats = tr.transition_computation_kmeans(
            activity=target_activity,
            scaler=scaler,
            threshold=threshold,
            k=k_value,
            file_path=file_path,
            enable_plots=enable_plots
        )
    
    print("\n== Initial Probabilities ==")
    print(df_initial)
    print("\n== Transition Matrix Columns ==")
    print(df_transition.columns)
    
    return df_transition, df_probs, df_initial, sequence_stats

def segment_test_data(test_activity, features, scaler, new_threshold):
    """
    Segment test data for all users.
    
    Returns:
        tuple: (rsv_df, lsv_users)
    """
    files = config.get_all_files_from_basepath(base_path=config.TEST_DATA_DIR)
    for file in files: 
        print(f"{file} \n")
    
    rsv_df = pd.DataFrame()
    lsv_users = {}
    
    for file in files: 
        username = str(file).split("/")[-1].split("_")[0]
        print(f"===== Segmentating user: {username}, test_activity: {test_activity}, scaler: {scaler}, threshold: {new_threshold} =====")
        rsv_user, lsv_user = vm.segment_user(username=username, file_path=file, features=features, 
                                            activity=test_activity, scaler=scaler, threshold=new_threshold)
        rsv_df = pd.concat([rsv_df, rsv_user], ignore_index=True)
        lsv_users[username] = lsv_user
    
    print("\n== RSV DataFrame: ==")
    print(rsv_df)
    
    for x in range(1, len(rsv_df['LogNumber'].unique()) + 1):
        print(f"LogNumber: {x}, Total number of segments: {len(rsv_df[rsv_df['LogNumber'] == x])}")
    
    return rsv_df, lsv_users

def classify_test_points(rsv_df, processed_data_path, result_directory, target_activity, 
                         threshold, scaler, clustering_algorithm, eps, min_samples, k_value, feature_len):
    """
    Classify test points and create log dataframe.
    
    Returns:
        tuple: (log_df, class_stats, class_threshold)
    """
    file_path = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}"
    if clustering_algorithm == "dbscan":
        df_dir = file_path + f"/all_points/dbscan_{target_activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv"
    elif clustering_algorithm == "kmeans":
        df_dir = file_path + f"/all_points/kmeans_{target_activity}_{threshold}_{scaler}_k{k_value}.csv"
    
    df = pd.read_csv(df_dir)
    
    # Debug: check what columns exist
    print(f"DEBUG: Loaded CSV columns: {df.columns.tolist()}")
    print(f"DEBUG: CSV shape: {df.shape}")
    
    # Check if Label column exists
    if 'Label' not in df.columns:
        raise KeyError(f"Label column not found in {df_dir}. Available columns: {df.columns.tolist()}")
    
    # Drop Adjective column if it exists
    if 'Adjective' in df.columns:
        df = df.drop(columns=['Adjective'])
    
    df = df.drop_duplicates()
    
    feature_columns = df.columns[-feature_len - 1: -1]
    print(f"DEBUG: feature_columns for classification: {feature_columns.tolist()}")
    
    df_inliers = df[df["Label"] != -1]
    print(f"DEBUG: df_inliers shape: {df_inliers.shape}, unique labels: {df_inliers['Label'].unique()}")
    
    if len(df_inliers) == 0:
        raise ValueError(f"No inliers found! All {len(df)} points are outliers (Label=-1). "
                        f"Try different clustering parameters or threshold.")
    
    class_stats = vm.compute_class_stats(df=df_inliers, feature_columns=feature_columns, label_column="Label", epsilon=1e-6)
    class_threshold = vm.compute_threshold(df=df_inliers, feature_columns=feature_columns, class_stats=class_stats, quantile=0.01)
    
    print("class_threshold: ", class_threshold)
    
    i = 0
    print("\n== Classify New Points ==")
    prev_label = None
    for index, row in rsv_df.iterrows():
        point = row[-feature_len:].to_numpy()
        print(f"Username {row['Username']}")
        label = vm.classify_point(point, class_stats, threshold=class_threshold)
        
        if index != 0 and row['LogNumber'] != prev_label:
            i = 1
        else: 
            i += 1
        
        prev_label = row['LogNumber']
        print(f"LogNumber: {row['LogNumber']} - Elementary Action: {i} - Classified as: {label}\n")
    
    log_df = pd.DataFrame({
        'Username': rsv_df['Username'],
        'LogNumber': rsv_df['LogNumber'],
        'Label': [vm.classify_point(row[-feature_len:].to_numpy(), class_stats, threshold=class_threshold) 
                 for _, row in rsv_df.iterrows()]
    })
    
    print("log_df: \n")
    print(log_df)
    
    return log_df, class_stats, class_threshold

def compute_user_states_and_probabilities(lsv_users, log_df, adj_df, df_probs, df_initial, 
                                         sequence_stats, ghost_exp, test_activity):
    """
    Compute user sequences and probabilities using Markov Chain.
    
    Returns:
        DataFrame: Results with probabilities and ghost counts
    """
    # Compute lsv mean for each user
    lsv_mean = ah.calculate_segment_means(lsv_users)
    
    # Compute user states (label, adjective)
    df_mean, _ = vm.compute_user_state(lsv_mean, log_df, adj_df, to_round=True)
    
    print("\n== User States DataFrame ==")
    print(df_mean)
    
    # Process each user
    all_users_results = {}
    all_df_sequence_results = []
    
    avg_sequence_length = sequence_stats['avg_states_per_sequence']
    print(f"\nAverage sequence length from training data: {avg_sequence_length}")
    
    n_states = len(df_probs.columns) - 1
    
    for username in df_mean['Username'].unique():
        print(f"\n{'='*60}")
        print(f"PROCESSING USER: {username}")
        print(f"{'='*60}")
        
        # Filter df_mean for the specific user
        df_mean_user = df_mean[df_mean['Username'] == username].copy()
        
        # Process sequences for this user
        user_ghost_end_results = []
        
        states = len(df_probs.columns) - 1
        
        print(f"States in the model (excluding ghost): {states}")
        
        for log_number, group in df_mean_user.groupby("LogNumber"):
            print(f"\n== GHOST STATE: Processing {username} - LogNumber: {log_number} ==")
            prob, N_states, ghost_count = vm.sequence_probability_ghost_state_end(
                group_df=group, 
                prob_matrix_df=df_probs, 
                initial_probs_df=df_initial, 
                states=states,
                exp=ghost_exp, 
                avg_sequence_length=n_states
            )
            user_ghost_end_results.append((username, test_activity, log_number, N_states, ghost_count, prob))
        
        # Create the DataFrame for the user
        df_sequence_user = pd.DataFrame(
            user_ghost_end_results, 
            columns=["Username", "TestActivity", "LogNumber", "N_states", "Ghost_Count", "Sequence_Probability"], 
            index=None
        )
        
        # Compute the metrics for the user
        df_sequence_user["Normalized"] = (df_sequence_user["Sequence_Probability"] / df_sequence_user["Sequence_Probability"].sum()).replace(np.nan, "Nan")
        df_sequence_user["LogProb"] = np.abs(np.log(df_sequence_user["Sequence_Probability"].replace(0, np.inf)))
        df_sequence_user["Length"] = df_mean_user.groupby("LogNumber").size().values
        df_sequence_user["AvgLogProb"] = (df_sequence_user["LogProb"] / df_sequence_user["Length"]).replace(np.nan, np.inf)
        
        # Save the result for the user
        all_users_results[username] = df_sequence_user
        all_df_sequence_results.append(df_sequence_user)
    
    # Combine the results in a single DataFrame
    df_sequence_ghost_end = pd.concat(all_df_sequence_results, ignore_index=True)
    print("\n== All Users Results ==")
    print(df_sequence_ghost_end)
    
    return df_sequence_ghost_end

def save_results(df_sequence_ghost_end, features, target_activity, test_activity, threshold, 
                new_threshold, clustering_algorithm, eps, min_samples, k_value, scaler, ghost_exp):
    """
    Save final results to CSV file.
    """
    columns = ["Username", "Features", "Target_Activity", "TestActivity", "Target_Threshold", 
              "Test_Threshold", "Clustering", "eps", "min_samples", "K", "scaler", "LogNumber", 
              "N_states", "Ghost_Count", "Ghost_Exp", "Concat", "AvgLogProb", "L"]
    
    if features == config.ROTATION_FEATURES: 
        feat = "rotation"
    elif features == config.POSITION_FEATURES:
        feat = "position"
    else:
        feat = "rotation_position"
    
    rows_to_add = []
    
    eps_val = "None" if clustering_algorithm == "kmeans" else eps
    min_samples_val = "None" if clustering_algorithm == "kmeans" else min_samples
    k_val = "None" if clustering_algorithm == "dbscan" else k_value
    
    for index, row in df_sequence_ghost_end.iterrows():
        rows_to_add.append({
            "Username": row['Username'],
            "Features": feat,
            "Target_Activity": target_activity,
            "TestActivity": test_activity,
            "Target_Threshold": threshold,
            "Test_Threshold": new_threshold,
            "Clustering": clustering_algorithm,
            "eps": eps_val,
            "min_samples": min_samples_val,
            "K": k_val,
            "scaler": scaler,
            "LogNumber": int(row["LogNumber"]),
            "L": "None",
            "N_states": row["N_states"],
            "Ghost_Count": row["Ghost_Count"],
            "Ghost_Exp": ghost_exp,
            "Concat": "markov_chain",
            "AvgLogProb": (row["AvgLogProb"] if not pd.isna(row["AvgLogProb"]) else np.inf)
        })
    
    df_save = pd.DataFrame(rows_to_add, columns=columns)
    
    path = config.RESULTS_DIR + f"/recognition_results/results_gen3.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    file_exists = os.path.exists(path)
    write_header = not file_exists
    df_save.to_csv(path, header=write_header, mode='a', index=False)
    
    print(df_save)
    
    # Check for and remove duplicates from the CSV file
    df_loaded = pd.read_csv(path)
    duplicates_count = df_loaded.duplicated().sum()
    
    if duplicates_count > 0:
        # Remove duplicates and save back
        df_cleaned = df_loaded.drop_duplicates()
        df_cleaned.to_csv(path, index=False)
        print(f"\nRemoved {duplicates_count} duplicate rows from results file.")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Results saved to: ", path)
    print("=" * 60)
    
    return df_save, path

def run_model_pipeline(
    target_activity="sphereActivity",
    test_activity="sphereActivity",
    threshold=0.7,
    scaler="standard",
    feature_mode="both",
    clustering_algorithm="kmeans",
    k_value=10,
    eps=0.25,
    min_samples=11,
    ghost_exp=4,
    enable_plots=False
):
    """
    Run the complete model pipeline with specified hyperparameters using Markov Chain.
    
    Args:
        target_activity: Activity to train on (default: "sphereActivity")
        test_activity: Activity to test on (default: "sphereActivity")
        threshold: Segmentation threshold for both training and testing (default: 0.7)
        scaler: Scaler type - "standard", "minmax", etc. (default: "standard")
        feature_mode: Feature mode - "rotation", "position", or "both" (default: "both")
        clustering_algorithm: Clustering algorithm - "kmeans" or "dbscan" (default: "kmeans")
        k_value: Number of clusters for KMeans (default: 10)
        eps: Epsilon parameter for DBSCAN (default: 0.25)
        min_samples: Minimum samples for DBSCAN (default: 11)
        ghost_exp: Ghost exponent parameter (default: 4)
        enable_plots: Whether to enable plotting (default: False)
        
    Returns:
        tuple: (df_save, path) - Results dataframe and save path
    """
    
    # Setup features and paths
    features, segmentation_result, histogram_dir, feature_len = setup_features_and_paths(feature_mode)
    processed_data_path = config.PROCESSED_DATA_DIR
    
    # Perform segmentation
    rsv_df, lsv_all_user, rsv_path = perform_segmentation(
        target_activity, features, threshold, scaler, segmentation_result, histogram_dir, enable_plots
    )
    
    # Prepare clustering data
    df, df_origin = prepare_clustering_data(rsv_path)
    
    # Perform clustering
    df_origin_labeled, df_no_outliers, unique_labels, result_directory = perform_clustering(
        df, df_origin, clustering_algorithm, k_value, eps, min_samples,
        target_activity, threshold, scaler, processed_data_path, enable_plots, feature_len
    )
    
    # Get best labels for adjectives
    print(f"DEBUG: df_origin_labeled columns: {df_origin_labeled.columns.tolist()}")
    print(f"DEBUG: df_origin_labeled shape: {df_origin_labeled.shape}")
    
    if 'Label' not in df_origin_labeled.columns:
        raise KeyError(f"'Label' column not found in df_origin_labeled after clustering. "
                      f"Available columns: {df_origin_labeled.columns.tolist()}")
    
    best_labels = df_origin_labeled['Label'].values
    print(f"DEBUG: best_labels length: {len(best_labels)}, unique values: {np.unique(best_labels)}")
    
    # Create substantives and adjectives
    df_substantives, df_adjectives, adj_dir = create_substantives_and_adjectives(
        df_no_outliers, unique_labels, df_origin_labeled, lsv_all_user, best_labels,
        result_directory, target_activity, threshold, scaler, processed_data_path,
        clustering_algorithm, eps, min_samples, k_value
    )
    
    # Compute transitions (Markov Chain)
    df_transition, df_probs, df_initial, sequence_stats = compute_transitions(
        clustering_algorithm, processed_data_path, result_directory,
        target_activity, scaler, threshold, k_value, eps, min_samples, enable_plots
    )
    
    # Segment test data (using same threshold as training)
    rsv_df_test, lsv_users = segment_test_data(test_activity, features, scaler, threshold)
    
    # Classify test points
    log_df, class_stats, class_threshold = classify_test_points(
        rsv_df_test, processed_data_path, result_directory, target_activity,
        threshold, scaler, clustering_algorithm, eps, min_samples, k_value, feature_len
    )
    
    # Load adjectives dataframe
    adj_df = pd.read_csv(adj_dir)
    
    # Compute sequences and probabilities using Markov Chain
    df_sequence_ghost_end = compute_user_states_and_probabilities(
        lsv_users, log_df, adj_df, df_probs, df_initial,
        sequence_stats, ghost_exp, test_activity
    )
    
    # Save results
    df_save, path = save_results(
        df_sequence_ghost_end, features, target_activity, test_activity,
        threshold, threshold, clustering_algorithm, eps, min_samples,
        k_value, scaler, ghost_exp
    )
    
    return df_save, path


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run model pipeline using Markov Chain')
    parser.add_argument('-p', '--plot', action='store_true', help='Enable plotting')
    parser.add_argument('--target-activity', type=str, default="sphereActivity", help='Target activity')
    parser.add_argument('--test-activity', type=str, default="sphereActivity", help='Test activity')
    parser.add_argument('--threshold', type=float, default=0.7, help='Segmentation threshold (used for both training and testing)')
    parser.add_argument('--scaler', type=str, default="standard", help='Scaler type')
    parser.add_argument('--feature-mode', type=str, default="both", 
                       choices=["rotation", "position", "both"], help='Feature mode')
    parser.add_argument('--clustering', type=str, default="kmeans", 
                       choices=["kmeans", "dbscan"], help='Clustering algorithm')
    parser.add_argument('-K', type=int, default=10, help='Number of clusters for KMeans')
    parser.add_argument('--eps', type=float, default=0.25, help='Epsilon for DBSCAN')
    parser.add_argument('--min-samples', type=int, default=11, help='Min samples for DBSCAN')
    parser.add_argument('--ghost-exp', type=int, default=4, help='Ghost exponent parameter')
    
    args = parser.parse_args()
    
    # Run the pipeline with parsed arguments
    run_model_pipeline(
        target_activity=args.target_activity,
        test_activity=args.test_activity,
        threshold=args.threshold,
        scaler=args.scaler,
        feature_mode=args.feature_mode,
        clustering_algorithm=args.clustering,
        k_value=args.K,
        eps=args.eps,
        min_samples=args.min_samples,
        ghost_exp=args.ghost_exp,
        enable_plots=args.plot
    )
