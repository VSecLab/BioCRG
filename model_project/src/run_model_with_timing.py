#!/usr/bin/env python3
"""
Script che esegue tutto il codice del notebook model.ipynb e misura il tempo di esecuzione
"""

import time
import os
import math
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

def main():
    print("=" * 80)
    print("INIZIO ESECUZIONE DEL MODELLO")
    print("=" * 80)
    
    # Start timing
    start_time = time.time()
    
    try:
        # ===== STEP 0: Initial Parameters Selection =====
        print("\n>>> STEP 0: Initial Parameters Selection")
        step_start = time.time()
        
        TARGET_ACTIVITY = "ladderActivity"
        TEST_ACTIVITY = "ladderActivity"
        K_VALUE = 8

        target_activity = TARGET_ACTIVITY 
        threshold = 0.7
        scaler = "standard"

        # both rotation and position
        features = config.FEATURES
        segmentation_result = config.BOTH_SEGMENTATION_DIR 
        processed_data_path = config.PROCESSED_DATA_DIR
        histogram_dir = config.BOTH_HISTOGRAMS_DIR

        feature_len = len(features) - 2 
        clustering_algorithm = "kmeans"
        
        step_time = time.time() - step_start
        print(f"    Completato in {step_time:.2f} secondi")

        # ===== STEP 1: Target Activity Segmentation =====
        print("\n>>> STEP 1: Target Activity Segmentation")
        step_start = time.time()
        
        rsv_df, lsv_all_user = sg.segment_all_users(
                    activity=target_activity,
                    features=features,
                    threshold=threshold,
                    scaler=scaler,
                    filepath=segmentation_result
                )

        print(f"RSV DataFrame shape: {rsv_df.shape}")

        os.makedirs(segmentation_result + f"/segments_number/{target_activity}", exist_ok=True)
        os.makedirs(histogram_dir + f"/{target_activity}", exist_ok=True)

        plot.plot_histogram_of_segments(
            filepath=segmentation_result + f"/segments_number/{target_activity}/{scaler}/segmentation_results_{target_activity}_{threshold}_{scaler}.csv",
            savepath=histogram_dir + f"/{target_activity}")
        
        step_time = time.time() - step_start
        print(f"    Completato in {step_time:.2f} secondi")

        # ===== STEP 2: Elementary Actions Clustering =====
        print("\n>>> STEP 2: Elementary Actions Clustering")
        step_start = time.time()
        
        rsv_path = segmentation_result + f"/right_singular_vector/{target_activity}/{scaler}/rsv_{target_activity}_{threshold}_{scaler}.csv"
        
        print(f"===== Processing activity: {target_activity}, threshold: {threshold}, scaler: {scaler} =====\n")

        df_origin = pd.read_csv(rsv_path, index_col=None)

        df = df_origin.drop(columns=['Username', "Adjective", 'LogNumber'])
        df = df.dropna()
        df = df.drop_duplicates()

        print("=================================================\n")
        
        # Grid search per iperparametri (solo per kmeans in questo caso)
        if clustering_algorithm == "kmeans":
            print("== KMEANS Clustering Results ==\n")
            n_cols = len(df.columns)
            length = len(df)
            print(f"Number of columns: {n_cols}\nLength of DataFrame: {length}\n")
            km.grid(df)
        
        # Impostazione iperparametri
        if clustering_algorithm == "dbscan":
            eps = 0.25
            min_samples = 11
        elif clustering_algorithm == "kmeans":
            K = K_VALUE
        
        # Esecuzione clustering
        if clustering_algorithm == "kmeans":
            best_labels = km.kmeans_with_outlier_detection(df=df, K=K)
            df['Label'] = best_labels

            feature_cols = df_origin.columns[-18:].to_list()

            df_origin = df_origin.merge(
                df,
                on=feature_cols,
                how="left"
            )
            result_directory = "kmeans_results"
            kmeans_dir = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}/all_points/"
            os.makedirs(kmeans_dir, exist_ok=True)

            save_dir = kmeans_dir + f"kmeans_{target_activity}_{threshold}_{scaler}_k{K}.csv"

            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            df_origin.to_csv(save_dir, index=False)

            df_no_outliers = df.drop(df[df['Label'] == -1].index)
            unique_labels = df_no_outliers['Label'].unique()

            print(f"Number of unique labels (excluding outliers): {len(unique_labels)}\n")
        
        step_time = time.time() - step_start
        print(f"    Completato in {step_time:.2f} secondi")

        # ===== STEP 2.4: Substantives =====
        print("\n>>> STEP 2.4: Substantives Generation")
        step_start = time.time()
        
        df_substantives = db.get_substantives_dataframe(df=df_no_outliers, unique_labels=unique_labels)
        print(f"\nSubstantives DataFrame shape: {df_substantives.shape}")
        
        if clustering_algorithm == "kmeans":
            save_dir = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}/substantives/substantives_{target_activity}_{threshold}_{scaler}_k{K}.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        df_substantives.to_csv(save_dir, index=False)
        
        step_time = time.time() - step_start
        print(f"    Completato in {step_time:.2f} secondi")

        # ===== STEP 2.5: Adjectives =====
        print("\n>>> STEP 2.5: Adjectives Generation")
        step_start = time.time()
        
        # Create LSV clusters dictionary
        df_new = df_origin.drop(columns=['Adjective']).drop_duplicates()

        # Obtain a dict where the key is the cluster label and the value are lsv
        lsv_clusters_dict = ah.adjectives_dict(best_labels, lsv_all_user, df_new)

        # Obtain a dict where the key is the cluster label and the value are segments of lsv
        clusters_dict = ah.adjectives_segmentation(lsv_clusters_dict)

        # Obtain a DataFrame with the label and the adjective(s) mean value for each segment
        df_adjectives = ah.generalize_adjectives(clusters=clusters_dict)

        print(f"\nAdjectives DataFrame shape: {df_adjectives.shape}")

        save_dir = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}/adjectives/"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        if clustering_algorithm == "kmeans":
            adj_dir = save_dir + f"adjectives_{target_activity}_{threshold}_{scaler}_k{K}.csv"

        df_adjectives.to_csv(adj_dir, index=False)
        
        step_time = time.time() - step_start
        print(f"    Completato in {step_time:.2f} secondi")

        # ===== STEP 3: Markov Chain =====
        print("\n>>> STEP 3: Markov Chain Transitions")
        step_start = time.time()
        
        if clustering_algorithm == "kmeans":
            df_transition, df_probs, df_initial, sequence_stats = tr.transition_computation_kmeans(
                    activity=target_activity,
                    scaler=scaler,
                    threshold=threshold,
                    k=K,
                    file_path=processed_data_path + f"/{result_directory}/{target_activity}/{scaler}"
                )

        print(f"Initial probabilities shape: {df_initial.shape}")
        print(f"Transition matrix shape: {df_transition.shape}")
        
        step_time = time.time() - step_start
        print(f"    Completato in {step_time:.2f} secondi")

        # ===== MODEL VALIDATION =====
        print("\n>>> MODEL VALIDATION")
        step_start = time.time()
        
        test_activity = TEST_ACTIVITY
        files = config.get_all_files_from_basepath(base_path=config.TEST_DATA_DIR)
        print(f"Test files found: {len(files)}")

        # Test Activity Segmentation
        new_threshold = threshold
        rsv_df = pd.DataFrame()
        lsv_users = {}

        for file in files: 
            username = str(file).split("/")[-1].split("_")[0]
            print(f"Segmentating user: {username}")
            rsv_user, lsv_user = vm.segment_user(username=username, file_path=file, features=features, activity=test_activity, scaler=scaler, threshold=new_threshold)
            rsv_df = pd.concat([rsv_df, rsv_user], ignore_index=True)
            lsv_users[username] = lsv_user

        print(f"\nRSV Test DataFrame shape: {rsv_df.shape}")
        
        step_time = time.time() - step_start
        print(f"    Test segmentation completata in {step_time:.2f} secondi")

        # ===== Maximum Likelihood Classifier =====
        print("\n>>> Maximum Likelihood Classifier")
        step_start = time.time()
        
        file_path = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}" 

        if clustering_algorithm == "kmeans":
            df_dir = file_path + f"/all_points/kmeans_{target_activity}_{threshold}_{scaler}_k{K}.csv"

        df = pd.read_csv(df_dir)
        df = df.drop(columns=['Adjective']).drop_duplicates()

        feature_columns = df.columns[-feature_len - 1: -1]  
        df_inliers = df[df["Label"] != -1]

        class_stats = vm.compute_class_stats(df=df_inliers, feature_columns=feature_columns, label_column="Label", epsilon=1e-6)
        class_threshold = vm.compute_threshold(df=df_inliers, feature_columns=feature_columns, class_stats=class_stats, quantile=0.01)

        print(f"Class threshold: {class_threshold}")

        # Classification
        prev_label = None
        i = 0
        classifications = []
        
        for index, row in rsv_df.iterrows():
            point = row[-feature_len:].to_numpy()
            label = vm.classify_point(point, class_stats, threshold=class_threshold)

            if index != 0 and row['LogNumber'] != prev_label:
                i = 1
            else: 
                i += 1

            prev_label = row['LogNumber']
            classifications.append(label)

        log_df = pd.DataFrame({
            'Username': rsv_df['Username'],
            'LogNumber': rsv_df['LogNumber'],
            'Label': classifications
        })

        print(f"Classification completed. Log DataFrame shape: {log_df.shape}")
        
        step_time = time.time() - step_start
        print(f"    Classification completata in {step_time:.2f} secondi")

        # ===== States Identification =====
        print("\n>>> States Identification")
        step_start = time.time()
        
        if clustering_algorithm == "kmeans":
            adj_df = pd.read_csv(file_path + f"/adjectives/adjectives_{target_activity}_{threshold}_{scaler}_k{K}.csv")

        # compute lsv mean for each of them 
        lsv_mean = ah.calculate_segment_means(lsv_users)

        # compute a DataFrame with the user state (label, adjective)
        df_mean = vm.compute_user_state(lsv_mean, log_df, adj_df)

        print(f"States DataFrame shape: {df_mean.shape}")
        
        step_time = time.time() - step_start
        print(f"    States identification completata in {step_time:.2f} secondi")

        # ===== Probability Computation =====
        print("\n>>> Probability Computation")
        step_start = time.time()
        
        ghost_exp = 4

        if clustering_algorithm == "kmeans":
            prob_matrix_dir = config.RESULTS_DIR + f"/transition_results/{target_activity}/{scaler}/transition_probabilities_{target_activity}_{threshold}_{scaler}_k{K}.csv"
            initial_probs_dir = config.RESULTS_DIR + f"/transition_results/{target_activity}/{scaler}/initial_probabilities_{target_activity}_{threshold}_{scaler}_k{K}.csv"

        os.makedirs(os.path.dirname(prob_matrix_dir), exist_ok=True)
        os.makedirs(os.path.dirname(initial_probs_dir), exist_ok=True)

        prob_matrix_df = pd.read_csv(prob_matrix_dir, index_col=0)
        initial_probs_df = pd.read_csv(initial_probs_dir)   

        # Process each user 
        all_users_results = {}
        all_df_sequence_results = []

        avg_sequence_length = sequence_stats['avg_states_per_sequence']
        print(f"Average sequence length from training data: {avg_sequence_length}")

        n_states = len(prob_matrix_df.columns) - 1

        for username in df_mean['Username'].unique():
            print(f"Processing user: {username}")
            
            # Filter df_mean for the specific user
            df_mean_user = df_mean[df_mean['Username'] == username].copy()

            # Process sequences for this user
            user_ghost_end_results = []
            states = len(prob_matrix_df.columns) - 1 
            
            for log_number, group in df_mean_user.groupby("LogNumber"):
                prob, N_states, ghost_count = vm.sequence_probability_ghost_state_end(
                    group_df=group, 
                    prob_matrix_df=prob_matrix_df, 
                    initial_probs_df=initial_probs_df, 
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
        df_sequence_ghost_end_all_users = pd.concat(all_df_sequence_results, ignore_index=True)
        df_sequence_ghost_end = df_sequence_ghost_end_all_users.copy()
        
        print(f"Final results shape: {df_sequence_ghost_end.shape}")
        
        step_time = time.time() - step_start
        print(f"    Probability computation completata in {step_time:.2f} secondi")

        # ===== Results Saving =====
        print("\n>>> Results Saving")
        step_start = time.time()
        
        columns = ["Username", "Features", "Target_Activity", "TestActivity", "Target_Threshold", "Test_Threshold", "Clustering","eps", "min_samples", "K", "scaler", "LogNumber", "N_states", "Ghost_Count", "Ghost_Exp", "Sequence_Probability", "Normalized", "LogProb", "Length", "AvgLogProb"]

        if features == config.ROTATION_FEATURES: 
            feat = "rotation"
        elif features == config.POSITION_FEATURES:
            feat = "position"
        else:
            feat = "rotation_position"
            
        # Create list to store all rows
        rows_to_add = []

        if clustering_algorithm == "dbscan":
            K = "None"
        elif clustering_algorithm == "kmeans":
            eps = "None"
            min_samples = "None"

        for index, row in df_sequence_ghost_end.iterrows():
            rows_to_add.append({
                "Username": row['Username'],
                "Features": feat,
                "Target_Activity": target_activity,
                "TestActivity": test_activity,
                "Target_Threshold": threshold,
                "Test_Threshold": new_threshold,
                "Clustering": clustering_algorithm,
                "eps": eps,
                "min_samples": min_samples,
                "K": K,
                "scaler": scaler,
                "LogNumber": int(row["LogNumber"]),
                "N_states": row["N_states"],
                "Ghost_Count": row["Ghost_Count"],
                "Ghost_Exp": ghost_exp,
                "Sequence_Probability": row["Sequence_Probability"],
                "Normalized": (row["Normalized"] if not pd.isna(row["Normalized"]) else 0.0),
                "LogProb": (row["LogProb"] if not pd.isna(row["LogProb"]) else np.inf),
                "Length": row["Length"],
                "AvgLogProb": (row["AvgLogProb"] if not pd.isna(row["AvgLogProb"]) else np.inf)
            })

        df_save = pd.DataFrame(rows_to_add, columns=columns)

        path = config.RESULTS_DIR + f"/recognition_results/results_oct2.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        file_exists = os.path.exists(path)
        write_header = not file_exists
        df_save.to_csv(path, header=write_header, mode='a', index=False)

        print(f"Results saved to: {path}")

        # Check for and remove duplicates from the CSV file
        df_loaded = pd.read_csv(path)
        duplicates_count = df_loaded.duplicated().sum()

        if duplicates_count > 0:
            # remove duplicates and save back 
            df_cleaned = df_loaded.drop_duplicates()
            df_cleaned.to_csv(path, index=False)
            print(f"Removed {duplicates_count} duplicates from results file")
        
        step_time = time.time() - step_start
        print(f"    Results saving completato in {step_time:.2f} secondi")

    except Exception as e:
        print(f"\nERRORE durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Calculate total execution time
    total_time = time.time() - start_time
    
    # Print final timing report
    print("\n" + "=" * 80)
    print("REPORT TEMPI DI ESECUZIONE")
    print("=" * 80)
    
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    print(f"Tempo totale di esecuzione: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
    print(f"Tempo totale in secondi: {total_time:.2f}")
    print(f"Tempo totale in minuti: {total_time/60:.2f}")
    
    if total_time > 3600:
        print(f"Tempo totale in ore: {total_time/3600:.2f}")
    
    print("\nEsecuzione completata con successo!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)