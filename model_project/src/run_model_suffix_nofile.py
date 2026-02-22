#!/usr/bin/env python3
"""
Script for running the complete model suffix pipeline.
All data is managed in-memory via variables instead of file I/O.
"""

import os
import sys
import argparse
import numpy as np 
import pandas as pd 
import kmeans as km
import segmentation as sg
import validation_model as vm
import dbscan_clustering as db
import adjectives_handle as ah
import suffix_tree.suffix_tree as st
import data_processing.src.plot as plot

from contextlib import contextmanager, nullcontext
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
    
    if enable_plots:
        os.makedirs(segmentation_result + f"/segments_number/{target_activity}", exist_ok=True)
        os.makedirs(histogram_dir + f"/{target_activity}", exist_ok=True)
        plot.plot_histogram_of_segments(
            filepath=segmentation_result + f"/segments_number/{target_activity}/{scaler}/segmentation_results_{target_activity}_{threshold}_{scaler}.csv",
            savepath=histogram_dir + f"/{target_activity}"
        )
    
    rsv_path = segmentation_result + f"/right_singular_vector/{target_activity}/{scaler}/rsv_{target_activity}_{threshold}_{scaler}.csv"
    
    return rsv_df, lsv_all_user, rsv_path


def prepare_clustering_data(rsv_df, feature_len):
    """
    Prepare data for clustering from rsv_df passed as variable.
    
    Args:
        rsv_df: DataFrame containing RSV data
        feature_len: Number of feature columns
    
    Returns:
        tuple: (df, df_origin)
    """
    df_origin = rsv_df.copy()
    df = df_origin.drop(columns=['Username', "Adjective", 'LogNumber'])
    df = df.dropna()
    df = df.drop_duplicates()
    
    return df, df_origin


def perform_clustering(df, df_origin, clustering_algorithm, k_value, eps, min_samples, 
                      target_activity, threshold, scaler, enable_plots, feature_len):
    """
    Perform clustering (DBSCAN or KMeans) on the data.
    
    Returns:
        tuple: (df_origin with labels, df_no_outliers, unique_labels, labeled_data_df)
    """
    # print("=================================================\n")
    
    if enable_plots:
        if clustering_algorithm == "dbscan": 
            # print("== DBSCAN Clustering Results ==\n")
            n_cols = len(df.columns)
            length = len(df)
            # print(f"Number of columns: {n_cols}\nLength of DataFrame: {length}\n")
            db.grid(df)
        elif clustering_algorithm == "kmeans":
            # print("== KMEANS Clustering Results ==\n")
            n_cols = len(df.columns)
            length = len(df)
            # print(f"Number of columns: {n_cols}\nLength of DataFrame: {length}\n")
            km.grid(df)
    
    if clustering_algorithm == "dbscan":
        best_labels = db.dbscan_clustering(df, eps=eps, min_samples=min_samples)
    elif clustering_algorithm == "kmeans":
        best_labels = km.kmeans_with_outlier_detection(df=df, K=k_value)
    
    # Debug: verify dimensions
    # print(f"DEBUG: df shape: {df.shape}, best_labels length: {len(best_labels)}")
    if len(best_labels) != len(df):
        raise ValueError(f"Labels length ({len(best_labels)}) doesn't match df length ({len(df)})")
    
    df['Label'] = best_labels
    
    feature_cols = df_origin.columns[-feature_len:].to_list()
    # print(f"DEBUG: feature_cols (last {feature_len}): {feature_cols}")
    df_origin = df_origin.merge(df, on=feature_cols, how="left")
    
    # Check for NaN labels after merge
    if df_origin['Label'].isna().any():
        nan_count = df_origin['Label'].isna().sum()
        # print(f"WARNING: {nan_count} rows have NaN labels after merge")
        # print(f"df_origin rows: {len(df_origin)}, df rows: {len(df)}")
        # Fill NaN with -1 (outlier label)
        df_origin['Label'] = df_origin['Label'].fillna(-1)
    
    df_no_outliers = df.drop(df[df['Label'] == -1].index)
    unique_labels = df_no_outliers['Label'].unique()
    
    # print(f"Number of unique labels (excluding outliers): {len(unique_labels)}\n")
    
    # Create labeled_data_df for later use (contains all points with labels)
    labeled_data_df = df_origin.copy()
    
    return df_origin, df_no_outliers, unique_labels, labeled_data_df


def create_substantives_and_adjectives(df_no_outliers, unique_labels, df_origin, lsv_all_user, 
                                       best_labels):
    """
    Create substantives and adjectives dataframes.
    
    Returns:
        tuple: (df_substantives, df_adjectives)
    """
    df_substantives = db.get_substantives_dataframe(df=df_no_outliers, unique_labels=unique_labels)
    # print("\n== Substantives DataFrame ==\n")
    # print(df_substantives)
    
    df_new = df_origin.drop(columns=['Adjective']).drop_duplicates()
    lsv_clusters_dict = ah.adjectives_dict(best_labels, lsv_all_user, df_new)
    clusters_dict = ah.adjectives_segmentation(lsv_clusters_dict)
    df_adjectives = ah.generalize_adjectives(clusters=clusters_dict)
    
    # print("\n== Adjectives DataFrame ==\n")
    # print(df_adjectives)
    
    return df_substantives, df_adjectives


def compute_sequences_from_dataframes(adj_df, activity_df):
    """
    Compute sequences from adjectives and activity dataframes.
    Replaces the file-based extract_activity_sequences from use_suffix_tree.py
    
    Args:
        adj_df: DataFrame with adjectives
        activity_df: DataFrame with all points and labels
    
    Returns:
        list: Lista di liste, dove ogni lista interna rappresenta una singola attività
    """
    def find_closest_adjective(adj_df: pd.DataFrame, label, target_adj):
        candidates = adj_df[adj_df['Label'] == label]
        same_sign = candidates[candidates['Adjective'] * target_adj >= 0]
        if same_sign.empty:
            return None
        closest = same_sign.iloc[(same_sign['Adjective'] - target_adj).abs().argsort()].iloc[0]
        return (label, closest['Adjective'])
    
    # Raggruppa per Username e LogNumber
    grouped = activity_df.groupby(['Username', 'LogNumber'])
    
    sequences = []
    
    for (username, log_number), group in grouped:
        group_sorted = group.sort_index()
        sequence = []
        
        for _, row in group_sorted.iterrows():
            label = int(row['Label'])
            adjective = float(row['Adjective'])
            
            if label == -1:
                matched_adjective = None
            else:
                matched_state = find_closest_adjective(adj_df, label, adjective)
                if matched_state:
                    matched_adjective = matched_state[1]
                else:
                    matched_adjective = None
            
            sequence.append((matched_adjective, label))
        
        sequences.append(sequence)
    
    return sequences


def assign_symbolic_states(sequences):
    """
    Assegna un valore simbolico a ogni stato unico trovato nelle sequenze.
    
    Args:
        sequences: Lista di sequenze, dove ogni sequenza contiene tuple (adjective, label)
    
    Returns:
        tuple: (state_mapping, symbolic_sequences)
    """
    unique_states = set()
    for sequence in sequences:
        for state in sequence:
            unique_states.add(state)
    
    sorted_states = sorted(unique_states, key=lambda x: (x[1] if x[1] is not None else -2, 
                                                          x[0] if x[0] is not None else float('-inf')))
    
    state_mapping = {}
    for idx, state in enumerate(sorted_states):
        state_mapping[state] = f"c{idx}"
    
    symbolic_sequences = []
    for sequence in sequences:
        symbolic_seq = [state_mapping[state] for state in sequence]
        symbolic_sequences.append(symbolic_seq)
    
    return state_mapping, symbolic_sequences


def convert_sequences_with_mapping(state_mapping, user_sequences):
    """
    Converte le sequenze utilizzando il mapping degli stati.
    
    Args:
        state_mapping: dizionario {(adjective, label): 'c0', 'c1', ...}
        user_sequences: dizionario {username: {LogNumber1: [], LogNumber2: []}}
    
    Returns:
        dict: dizionario con la stessa struttura ma con le sequenze convertite in simboli
    """
    converted_sequences = {}
    
    for username, log_dict in user_sequences.items():
        converted_sequences[username] = {}
        
        for log_number, sequence in log_dict.items():
            converted_seq = [state_mapping[state] for state in sequence]
            converted_sequences[username][log_number] = converted_seq
    
    return converted_sequences


def create_suffix_tree(clustering_algorithm, adj_df, labeled_data_df, 
                      l_value, pmin=0.000001, gamma_min=0.0, eps_suffix=0.0):
    """
    Create suffix tree from in-memory dataframes.
    
    Returns:
        tuple: (suffix_tree, state_mapping) or (None, None) for dbscan
    """
    if clustering_algorithm == "kmeans":
        # Compute sequences from dataframes instead of files
        sequences = compute_sequences_from_dataframes(adj_df, labeled_data_df)
        state_mapping, symbolic_sequences = assign_symbolic_states(sequences)
        
        # print("\n" + "=" * 60)
        # print("Creating Suffix Tree...")
        # print("=" * 60)
        suffix_tree = st.create_tree(sequences=symbolic_sequences, alphabet=list(state_mapping.values()), 
                                    L=l_value, pmin=pmin, gamma_min=gamma_min, eps=eps_suffix)
        # print("\n" + "=" * 60)
        
        return suffix_tree, state_mapping
    
    return None, None


def segment_test_data(test_activity, features, scaler, new_threshold, process_id = None):
    """
    Segment test data for all users.
    
    Returns:
        tuple: (rsv_df, lsv_users)
    """
    files = config.get_all_files_from_basepath(base_path=config.TEST_DATA_DIR)
    #for file in files: 
        # print(f"{file} \n")

    
    rsv_df = pd.DataFrame()
    lsv_users = {}
    
    for file in files: 
        username = str(file).split("/")[-1].split("_")[0]
        #print(f"===== Segmentating user: {username}, test_activity: {test_activity}, scaler: {scaler}, threshold: {new_threshold} =====")
        results = vm.segment_user(username=username, file_path=file, features=features, 
                                            activity=test_activity, scaler=scaler, threshold=new_threshold)
        if results is None:
            print(f"Skipping user {username} - segment_user returned None")
            continue
        
        rsv_user, lsv_user = results
        rsv_df = pd.concat([rsv_df, rsv_user], ignore_index=True)
        lsv_users[username] = lsv_user
    
    ## print("\n== RSV DataFrame: ==")
    # print(rsv_df)
    

    return rsv_df, lsv_users


def classify_test_points(rsv_df, labeled_data_df, feature_len):
    """
    Classify test points using in-memory labeled_data_df.
    
    Args:
        rsv_df: Test data RSV DataFrame
        labeled_data_df: Training data with labels (from clustering)
        feature_len: Number of feature columns
    
    Returns:
        tuple: (log_df, class_stats, class_threshold)
    """
    df = labeled_data_df.copy()
    
    # Debug: check what columns exist
    # print(f"DEBUG: Loaded CSV columns: {df.columns.tolist()}")
    # print(f"DEBUG: CSV shape: {df.shape}")
    
    # Check if Label column exists
    if 'Label' not in df.columns:
        raise KeyError(f"Label column not found. Available columns: {df.columns.tolist()}")
    
    # Drop Adjective column if it exists
    if 'Adjective' in df.columns:
        df = df.drop(columns=['Adjective'])
    
    df = df.drop_duplicates()
    
    feature_columns = df.columns[-feature_len - 1: -1]
    # print(f"DEBUG: feature_columns for classification: {feature_columns.tolist()}")
    
    df_inliers = df[df["Label"] != -1]
    # print(f"DEBUG: df_inliers shape: {df_inliers.shape}, unique labels: {df_inliers['Label'].unique()}")
    
    if len(df_inliers) == 0:
        raise ValueError(f"No inliers found! All {len(df)} points are outliers (Label=-1). "
                        f"Try different clustering parameters or threshold.")
    
    class_stats = vm.compute_class_stats(df=df_inliers, feature_columns=feature_columns, label_column="Label", epsilon=1e-6)
    class_threshold = vm.compute_threshold(df=df_inliers, feature_columns=feature_columns, class_stats=class_stats, quantile=0.01)
    
    # print("class_threshold: ", class_threshold)
    
    i = 0
    # print("\n== Classify New Points ==")
    prev_label = None
    for index, row in rsv_df.iterrows():
        point = row[-feature_len:].to_numpy()
        # print(f"Username {row['Username']}")
        label = vm.classify_point(point, class_stats, threshold=class_threshold)
        
        if index != 0 and row['LogNumber'] != prev_label:
            i = 1
        else: 
            i += 1
        
        prev_label = row['LogNumber']
        # print(f"LogNumber: {row['LogNumber']} - Elementary Action: {i} - Classified as: {label}\n")
    
    log_df = pd.DataFrame({
        'Username': rsv_df['Username'],
        'LogNumber': rsv_df['LogNumber'],
        'Label': [vm.classify_point(row[-feature_len:].to_numpy(), class_stats, threshold=class_threshold) 
                 for _, row in rsv_df.iterrows()]
    })
    
    # print("log_df: \n")
    # print(log_df)
    
    return log_df, class_stats, class_threshold


def compute_sequences_and_probabilities(lsv_users, log_df, adj_df, state_mapping, suffix_tree, 
                                       l_value, ghost_exp, test_activity):
    """
    Compute user sequences and probabilities using suffix tree.
    
    Returns:
        DataFrame: Results with probabilities and ghost counts
    """
    lsv_mean = ah.calculate_segment_means(lsv_users)
    df_mean, sequences_dict = vm.compute_user_state(lsv_mean, log_df, adj_df)
    
    # print(state_mapping)
    # print(sequences_dict)
    
    converted_sequences = convert_sequences_with_mapping(state_mapping, sequences_dict)
    
    # print("\n== Converted Sequences: ==\n")
    #for user, sequences in converted_sequences.items():
        # print(f"User: {user}")
        #for log_number, seq in sequences.items():
            # print(f"  LogNumber: {log_number} - Sequence: {seq}")
        # print()
    
    n_states = len(state_mapping) - 1
    all_users_results = {}
    all_df_sequence_results = []
    
    for username in converted_sequences.keys():
        user_ghost_end_results = []
        for log_number, seq in converted_sequences[username].items():
            # print(f"Username: {username} - LogNumber: {log_number} - Sequence: {seq}")
            self_info, N_states, ghost_count = st.PST_Probability_ghost(
                PST=suffix_tree,
                sequence=seq,
                L=l_value,
                states=n_states,
                ghost_exp=ghost_exp,
                avg_sequence_length=n_states
            )
            user_ghost_end_results.append((username, test_activity, log_number, N_states, ghost_count, self_info))
        
        df_sequence_user = pd.DataFrame(
            user_ghost_end_results, 
            columns=["Username", "TestActivity", "LogNumber", "N_states", "Ghost_Count", "AvgLogProb"], 
            index=None
        )
        
        all_users_results[username] = df_sequence_user
        all_df_sequence_results.append(df_sequence_user)
    
    df_sequence_ghost_end_all_users = pd.concat(all_df_sequence_results, ignore_index=True)
    df_sequence_ghost_end = df_sequence_ghost_end_all_users.copy()
    # print(df_sequence_ghost_end)
    
    return df_sequence_ghost_end

def save_results(df_sequence_ghost_end, features, target_activity, test_activity, threshold, 
                new_threshold, clustering_algorithm, eps, min_samples, k_value, scaler, l_value, 
                ghost_exp, pmin, gamma_min, eps_suffix):
    """
    Save final results to CSV file.
    """
    columns = ["Username", "Features", "Target_Activity", "TestActivity", "Target_Threshold", 
              "Test_Threshold", "Clustering", "eps", "min_samples", "K", "scaler", "pmin", 
              "gamma_min", "eps_suffix", "LogNumber", 
              "N_states", "Ghost_Count", "Ghost_Exp", "Concat", "L", "AvgLogProb"]
    
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
            "L": l_value,
            "N_states": row["N_states"],
            "Ghost_Count": row["Ghost_Count"],
            "Ghost_Exp": ghost_exp,
            "Concat": "suffix_tree",
            "AvgLogProb": (row["AvgLogProb"] if not pd.isna(row["AvgLogProb"]) else np.inf),
            "pmin": pmin,
            "gamma_min": gamma_min,
            "eps_suffix": eps_suffix
        })
    
    df_save = pd.DataFrame(rows_to_add, columns=columns)
    
    path = config.RESULTS_DIR + f"/recognition_results/results_gen_2.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    file_exists = os.path.exists(path)
    write_header = not file_exists
    df_save.to_csv(path, header=write_header, mode='a', index=False)
    
    # print(df_save)
    
    # print("\n" + "=" * 60)
    # print("Pipeline completed successfully!")
    # print("Results saved to: ", path)
    # print("=" * 60)
    
    return df_save, path

def create_results_dataframe(df_sequence_ghost_end, features, target_activity, test_activity, 
                             threshold, new_threshold, clustering_algorithm, eps, min_samples, 
                             k_value, scaler, l_value, ghost_exp, pmin, gamma_min, eps_suffix):
    """
    Create final results dataframe with all columns.
    """
    columns = ["Username", "Features", "Target_Activity", "TestActivity", "Target_Threshold", 
              "Test_Threshold", "Clustering", "eps", "min_samples", "K", "scaler", "pmin", 
              "gamma_min", "eps_suffix", "LogNumber", 
              "N_states", "Ghost_Count", "Ghost_Exp", "Concat", "L", "AvgLogProb"]
    
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
            "L": l_value,
            "N_states": row["N_states"],
            "Ghost_Count": row["Ghost_Count"],
            "Ghost_Exp": ghost_exp,
            "Concat": "suffix_tree",
            "AvgLogProb": (row["AvgLogProb"] if not pd.isna(row["AvgLogProb"]) else np.inf),
            "pmin": pmin,
            "gamma_min": gamma_min,
            "eps_suffix": eps_suffix
        })
    
    df_save = pd.DataFrame(rows_to_add, columns=columns)
    return df_save

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

def run_model_suffix_pipeline(
    target_activity="sphereActivity",
    test_activity="sphereActivity",
    threshold=0.7,
    scaler="standard",
    feature_mode="both",
    clustering_algorithm="kmeans",
    k_value=10,
    eps=0.25,
    min_samples=11,
    l_value=2,
    ghost_exp=4,
    pmin=0.000001,
    gamma_min=0.0,
    eps_suffix=0.0,
    enable_plots=False,
    process_id=None,
    save_output=True
):
    """
    Run the complete model suffix pipeline with specified hyperparameters.
    All data is managed in-memory via variables instead of file I/O.
    
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
        l_value: L parameter for suffix tree (default: 2)
        ghost_exp: Ghost exponent parameter (default: 4)
        pmin: Minimum probability for border expansion in suffix tree (default: 0.000001)
        gamma_min: Minimum transition probability in suffix tree (default: 0.0)
        eps_suffix: Epsilon parameter for significativity test in suffix tree (default: 0.0)
        enable_plots: Whether to enable plotting (default: False)
        save_output: Whether to save output to file (default: True)
    Returns:
        dict: Dictionary containing all pipeline results:
            - 'final_results': DataFrame with final probabilities
            - 'rsv_df': Training RSV data
            - 'lsv_all_user': Training LSV data
            - 'labeled_data': Training data with cluster labels
            - 'substantives': Substantives DataFrame
            - 'adjectives': Adjectives DataFrame
            - 'suffix_tree': The trained suffix tree (or None)
            - 'state_mapping': State mapping dictionary (or None)
            - 'test_rsv': Test RSV data
            - 'test_lsv': Test LSV data
            - 'log_df': Test data classifications
            - 'class_stats': Classification statistics
            - 'class_threshold': Classification threshold
            - 'save_path': Path where results were saved (if save_output=True)
    """
    
    # Usa suppress_stdout solo se process_id è diverso da 2 (o se è None)
    #should_suppress = process_id != 2 if process_id is not None else True


    # Setup features and paths
    features, segmentation_result, histogram_dir, feature_len = setup_features_and_paths(feature_mode)
    
    # Perform segmentation
    # print(f"===== Processing activity: {target_activity}, threshold: {threshold}, scaler: {scaler} =====\n")
    rsv_df, lsv_all_user, _ = perform_segmentation(
        target_activity, features, threshold, scaler, segmentation_result, histogram_dir, enable_plots
    )
    print(f"Process id: {process_id} - Completed segmentation for training data.")

    # Prepare clustering data (use in-memory rsv_df instead of reading from file)
    df, df_origin = prepare_clustering_data(rsv_df, feature_len)
    
    # Perform clustering
    df_origin_labeled, df_no_outliers, unique_labels, labeled_data_df = perform_clustering(
        df, df_origin, clustering_algorithm, k_value, eps, min_samples,
        target_activity, threshold, scaler, enable_plots, feature_len
    )
    print(f"Process id: {process_id} - Completed clustering for training data.")
    
    # Get best labels for adjectives
    #print(f"DEBUG: df_origin_labeled columns: {df_origin_labeled.columns.tolist()}")
    #print(f"DEBUG: df_origin_labeled shape: {df_origin_labeled.shape}")
    
    if 'Label' not in df_origin_labeled.columns:
        raise KeyError(f"'Label' column not found in df_origin_labeled after clustering. "
                    f"Available columns: {df_origin_labeled.columns.tolist()}")
    
    best_labels = df_origin_labeled['Label'].values
    #print(f"DEBUG: best_labels length: {len(best_labels)}, unique values: {np.unique(best_labels)}")
    
    # Create substantives and adjectives (no file saving)
    df_substantives, df_adjectives = create_substantives_and_adjectives(
        df_no_outliers, unique_labels, df_origin_labeled, lsv_all_user, best_labels
    )
    print(f"Process id: {process_id} - Completed creation of substantives and adjectives for training data.")
    # Create suffix tree (using in-memory dataframes instead of files)
    suffix_tree, state_mapping = create_suffix_tree(
        clustering_algorithm, df_adjectives, labeled_data_df,
        l_value, pmin, gamma_min, eps_suffix
    )

    print(f"Process id: {process_id} - Completed clustering and suffix tree creation for training data.")
    
    # Segment test data (using same threshold as training)
    results  = segment_test_data(test_activity, features, scaler, threshold, process_id=process_id)
    print(f"Process id: {process_id} - Completed segmentation of test data.")
    rsv_df_test, lsv_users = results


    # Classify test points (using in-memory labeled_data_df)
    log_df, class_stats, class_threshold = classify_test_points(
        rsv_df_test, labeled_data_df, feature_len
    )
    print(f"Process id: {process_id} - Classified test points for test data.")
    # Compute sequences and probabilities
    df_sequence_ghost_end = compute_sequences_and_probabilities(
        lsv_users, log_df, df_adjectives, state_mapping, suffix_tree,
        l_value, ghost_exp, test_activity
    )
    print(f"Process id: {process_id} - Computed sequences and probabilities for test data.")

    # Save results (optional)
    save_path = None
    if save_output:
        df_save, save_path = save_results(
            df_sequence_ghost_end, features, target_activity, test_activity,
            threshold, threshold, clustering_algorithm, eps, min_samples,
            k_value, scaler, l_value, ghost_exp, pmin, gamma_min, eps_suffix
        )  
    else:
        df_save = df_sequence_ghost_end
    

    df_save = create_results_dataframe(
        df_sequence_ghost_end, features, target_activity, test_activity,
        threshold, threshold, clustering_algorithm, eps, min_samples,
        k_value, scaler, l_value, ghost_exp, pmin, gamma_min, eps_suffix
    )
    print(f"Process id: {process_id} - Final results dataframe created with shape: {df_save.shape}")
    
    # Return all intermediate and final results
    results = {
        'final_results': df_save,
        'rsv_df': rsv_df,
        'lsv_all_user': lsv_all_user,
        'labeled_data': labeled_data_df,
        'substantives': df_substantives,
        'adjectives': df_adjectives,
        'suffix_tree': suffix_tree,
        'state_mapping': state_mapping,
        'test_rsv': rsv_df_test,
        'test_lsv': lsv_users,
        'log_df': log_df,
        'class_stats': class_stats,
        'class_threshold': class_threshold,
        'save_path': save_path
    }
    print(f"Process id: {process_id} - Pipeline completed successfully!")
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run model suffix pipeline')
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
    parser.add_argument('-L', type=int, default=2, help='L parameter for suffix tree')
    parser.add_argument('--ghost-exp', type=int, default=4, help='Ghost exponent parameter')
    parser.add_argument('--pmin', type=float, default=0.000001, help='Minimum probability for border expansion in suffix tree')
    parser.add_argument('--gamma-min', type=float, default=0.0, help='Minimum transition probability in suffix tree')
    parser.add_argument('--eps-suffix', type=float, default=0.0, help='Epsilon for significativity test in suffix tree')
    parser.add_argument('--no-save', action='store_true', help='Disable saving output to file')
    
    args = parser.parse_args()
    
    # Run the pipeline with parsed arguments
    results = run_model_suffix_pipeline(
        target_activity=args.target_activity,
        test_activity=args.test_activity,
        threshold=args.threshold,
        scaler=args.scaler,
        feature_mode=args.feature_mode,
        clustering_algorithm=args.clustering,
        k_value=args.K,
        eps=args.eps,
        min_samples=args.min_samples,
        l_value=args.L,
        ghost_exp=args.ghost_exp,
        pmin=args.pmin,
        gamma_min=args.gamma_min,
        eps_suffix=args.eps_suffix,
        enable_plots=args.plot,
        save_output=not args.no_save
    )
    
    print("\n" + "=" * 60)
    print("All results available in 'results' dictionary:")
    print(f"  - final_results: {results['final_results'].shape}")
    print(f"  - labeled_data: {results['labeled_data'].shape}")
    print(f"  - substantives: {results['substantives'].shape}")
    print(f"  - adjectives: {results['adjectives'].shape}")
    print(f"  - state_mapping: {len(results['state_mapping']) if results['state_mapping'] else 'None'} states")
    print("=" * 60)