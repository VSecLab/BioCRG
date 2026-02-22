import numpy as np
import pandas as pd

POS_TH = 0.2   
NEG_TH = -0.2


def calculate_segment_means(lsv_dict: dict):
    """
    The structure of the dictionary is as follows:
    {
        username: 
        {
            logNum_X: 
            {
                segNum_X: 
                    lsv_data, 
                    ...
            }, 
        ...
        }
    }

    Compute the mean of each segment vector (segNum_X) while maintaining the same dictionary structure.

    Args:
        lsv_dict (dict): dictionary with structure {username: {logNum_X: {segNum_Y: np.array([...])}}}

    Returns:
        dict: Dictionary with the same structure but with means instead of vectors
    """
    result_dict = {}
    
    for username, logs in lsv_dict.items():
        result_dict[username] = {}
        
        if logs is None:
            # print(f"WARNING: User {username} has None value in lsv_dict, skipping")
            continue

        for log_num, segments in logs.items():
            result_dict[username][log_num] = {}
            
            for seg_num, lsv_vector in segments.items():
                # Calcola la media del vettore
                result_dict[username][log_num][seg_num] = np.mean(lsv_vector)
    
    return result_dict

def adjectives_dict(best_labels: np.array, lsv_all_user: dict, df_origin: pd.DataFrame):
    """
    Creates a dictionary mapping each cluster label to its corresponding list of LSVs.
    The structure of the dictionary is as follows:
    {
        username: 
        {
            logNum_X: 
            {
                segNum_X: 
                    lsv_data, 
                    ...
            }, 
        ...
        }
    }

    Args:
        best_labels (np.ndnp.array): np.array of cluster labels for each RSV.
        lsv_all_user (dict): Dictionary where keys are usernames and values are dictionaries of log numbers mapping to segment numbers and their corresponding LSV vectors.
        df_origin (pd.DataFrame): Original DataFrame containing 'Username', 'LogNumber', and other relevant columns.

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of LSVs corresponding to that cluster.
    """
    lsv_clusters_dict = {}
    # print("Creating LSV clusters dictionary...")

    # Initialize dictionary with all unique cluster labels (including -1 for outliers)
    all_cluster_labels = np.unique(best_labels)
    for cluster_label in all_cluster_labels:
        lsv_clusters_dict[int(cluster_label)] = []

    # Map each RSV to its corresponding LSV using the clustering results
    for idx, (_, row) in enumerate(df_origin.iterrows()):
        username = row['Username']
        log_number = int(row['LogNumber'])
        cluster_label = int(row['Label'])
        
        # Create the log key as it appears in lsv_all_user
        log_key = f"logNum_{log_number}"
        
        # Find the corresponding LSV in lsv_all_user dictionary
        if username in lsv_all_user and log_key in lsv_all_user[username]:
            # Get the segment number for this RSV (we need to track which segment this RSV corresponds to)
            # Count how many segments from the same user and log we've seen so far
            user_log_segments = df_origin[(df_origin['Username'] == username) & (df_origin['LogNumber'] == log_number)]
            user_log_segments_before = user_log_segments[user_log_segments.index <= row.name]
            segment_number = len(user_log_segments_before)
            
            # Create the segment key as it appears in lsv_all_user
            seg_key = f"segNum_{segment_number}"
            
            if seg_key in lsv_all_user[username][log_key]:
                lsv_vector = lsv_all_user[username][log_key][seg_key]
                
                # Add metadata to identify the LSV
                lsv_info = {
                    'username': username,
                    'log_number': log_key,
                    'segment_number': seg_key,
                    'lsv_vector': lsv_vector
                }

                lsv_clusters_dict[cluster_label].append(lsv_info)
                #print(f"Added LSV for {username}, {log_key}, {seg_key} to cluster {cluster_label}")
            else:
               print(f"Warning: {seg_key} not found for {username}, {log_key}")

    return lsv_clusters_dict

def segment_lsv(lsv_vector: np.ndarray):
    """
    Segments a single LSV vector into four categories based on thresholds:
    - Slow Positive (0 < value < pos_th)
    - Fast Positive (value >= pos_th)
    - Slow Negative (neg_th < value < 0)
    - Fast Negative (value <= neg_th)

    Args:
        lsv_vector (np.ndnp.array): The LSV vector to segment.
        pos_th (float): Threshold for positive segmentation.
        neg_th (float): Threshold for negative segmentation.

    Returns:
        dict: A dictionary with keys 'slow_pos', 'fast_pos', 'slow_neg', 'fast_neg' and values as lists of segmented LSVs.
    """

    subsequences = {"slow_pos": [], "fast_pos": [], "slow_neg": [], "fast_neg": []}
    
    if len(lsv_vector) == 0:
        return subsequences

    # funzione per assegnare etichetta a un singolo valore
    def label_value(val):
        if val > 0:
            return "slow_pos" if val < POS_TH else "fast_pos"
        elif val < 0:
            return "slow_neg" if val > NEG_TH else "fast_neg"
        else:
            return None  # opzionale: zero ignorato

    # etichettatura del vettore
    labels = [label_value(v) for v in lsv_vector]

    # segmentazione in blocchi consecutivi
    start = 0
    current_label = labels[0]
    for i in range(1, len(lsv_vector)):
        if labels[i] != current_label:
            if current_label is not None:
                subsequences[current_label].append(lsv_vector[start:i])
            start = i
            current_label = labels[i]
    # aggiungi ultimo blocco
    if current_label is not None:
        subsequences[current_label].append(lsv_vector[start:])

    return subsequences

def adjectives_segmentation(lsv_clusters_dict: dict):
    """
    Segments LSVs in each cluster into four categories based on thresholds:
    - Slow Positive (0 < value < positive_threshold)
    - Fast Positive (value >= positive_threshold)
    - Slow Negative (negative_threshold < value < 0)
    - Fast Negative (value <= negative_threshold)

    Args:
        lsv_clusters_dict (dict): Dictionary where keys are cluster labels and values are lists of LSVs.

    Returns:
        dict: A dictionary with the same cluster labels, where each label maps to another dictionary containing
              the four segmented categories with their respective LSV segments.
    """
    
    clusters = {}
    
    for cluster_label, lsv_list in lsv_clusters_dict.items():
        if cluster_label == -1:
            continue
        if len(lsv_list) == 0:
            continue

        # Inizializza i 4 dizionari per ogni cluster
        clusters[cluster_label] = {
            "slow_pos": [],
            "slow_neg": [],
            "fast_pos": [],
            "fast_neg": []
        }

        for item in lsv_list: 
            lsv_vector = item['lsv_vector']

            segmented = segment_lsv(lsv_vector)
            
            # Append dei risultati ai rispettivi dizionari del cluster
            for adjective_type, segments in segmented.items():
                clusters[cluster_label][adjective_type].extend(segments)
    
    return clusters
            
def generalize_adjectives(clusters: dict):
    """
    Generalizes the segmented LSVs in each cluster by computing the mean vector for each adjective type.

    Args:
        clusters (dict): Dictionary where keys are cluster labels and values are dictionaries containing
                        lists of segmented LSVs for each adjective type.

    Returns:
        pd.DataFrame: DataFrame with columns 'Label' (cluster) and 'Adjective' (mean of means for each adjective type)
    """
    
    results = []
    
    for cluster_label, adjective_dict in clusters.items():
        for adjective_type, segments in adjective_dict.items():
            if len(segments) > 0:  # Solo se ci sono segmenti
                # Calcola la media per ogni segmento
                segment_means = [np.mean(segment) for segment in segments if len(segment) > 0]
                
                if len(segment_means) > 0:
                    # Calcola la media delle medie
                    mean_of_means = np.mean(segment_means)
                    
                    # Aggiungi al risultato
                    results.append({
                        'Label': cluster_label,
                        'Adjective': mean_of_means
                    })
    
    return pd.DataFrame(results)

            



            