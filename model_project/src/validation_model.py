import os
import numpy as np 
import pandas as pd
import segmentation as sg

from data_processing.src import config
from scipy.stats import multivariate_normal


def ghost_func(n: int, N: int, exp: int) -> float:
    return np.exp(- (n ** exp) / N)

def sequence_probability_ghost_state_end(group_df: pd.DataFrame, prob_matrix_df: pd.DataFrame, initial_probs_df: pd.DataFrame, exp: int = 4):
    group_df = group_df.sort_index()

    sequence = group_df['MatchedState'].tolist()

    N_states = len([s for s in sequence])
    ghost_state = "(-1, None)"
    ghost_count = sequence.count(ghost_state)

    # Verifica che tutti gli stati siano validi
    for state in sequence:
        if state not in prob_matrix_df.columns:
            print(f"State {state} not found in transition matrix.")
            return np.nan, N_states, ghost_count
        

    if N_states == ghost_count:
        print(f"All states are ghost states. Returning 0 probability.")
        return 0.0, N_states, ghost_count
    elif N_states == 1: 
        print(f"Only one state present. Returning 0 probability.")
        return 0.0, N_states, ghost_count

    # Calcolo della penalità da applicare nei salti ghost
    ghost_penalty = ghost_func(ghost_count, N_states, exp=exp)

    print(f"N_states: {N_states} - Ghost Count: {ghost_count} - Ghost Penalty: {ghost_penalty}")

    # Calcolo probabilità iniziale
    init_prob_map = dict(zip(initial_probs_df['State'], initial_probs_df['Initial_Probability']))
    prob = init_prob_map.get(sequence[0], 0)
    print(f"Initial Probability for {sequence[0]}: {prob}")

    if prob == 0:
        return 0.0, N_states, ghost_count
    
    #print(f"Initial Probability for {sequence[0]}: {prob}")

    # Probabilità di transizione
    for i in range(len(sequence) - 1):
        from_state = sequence[i]
        to_state = sequence[i + 1]
        trans_prob = prob_matrix_df.at[from_state, to_state]
        prob *= trans_prob
        print(f"probability from {from_state} to {to_state}: {trans_prob}, cumulative probability: {prob}")
        if prob == 0:
            print(f"Transition probability from {from_state} to {to_state} is zero, breaking the loop.")
            break
        #print(f"probability from {from_state} to {to_state}: {trans_prob}, cumulative probability: {prob}")
    #print(f"Ghost penalty: {ghost_penalty} - Probability: {prob} - Adjusted Probability: {prob * ghost_penalty}")
    return prob * ghost_penalty, N_states, ghost_count

def sequence_probability_ghost_state(group_df: pd.DataFrame, prob_matrix_df: pd.DataFrame, initial_probs_df: pd.DataFrame) -> float:
    group_df = group_df.sort_index()
    sequence = group_df['MatchedState'].tolist()

    # Verifica che tutti gli stati siano validi
    for state in sequence:
        if state not in prob_matrix_df.columns:
            print(f"State {state} not found in transition matrix.")
            return np.nan
    
    #print(sequence)

    N_states = len([s for s in sequence])
    ghost_state = "(-1, None)"
    ghost_count = sequence.count(ghost_state)

    # Calcolo della penalità da applicare nei salti ghost
    ghost_penalty = ghost_func(ghost_count, N_states)

    print(f"N_states: {N_states} - Ghost Count: {ghost_count} - Ghost Penalty: {ghost_penalty}")

    init_prob_map = dict(zip(initial_probs_df['State'], initial_probs_df['Initial_Probability']))
    first_state = sequence[0]
    prob = init_prob_map.get(first_state, 0.0)

    if prob == 0.0:
        return 0.0

    i = 0
    while i < len(sequence) - 1:
        current_state = sequence[i]
        next_state = sequence[i + 1]
        #print(f"Processing: {current_state} -> {next_state}")
        if current_state == ghost_state:
            if i != 0: 
                i += 1
                continue

        # Caso: transizione diretta S1 -> S2
        if next_state != ghost_state:
            if current_state not in prob_matrix_df.index or next_state not in prob_matrix_df.columns:
                return 0.0
            trans_prob = prob_matrix_df.at[current_state, next_state]
            prob *= trans_prob
            i += 1
        else:
            # Caso: S1 -> GS -> S2
            if i + 2 < len(sequence):
                after_ghost = sequence[i + 2]
                if after_ghost != ghost_state:
                    if current_state not in prob_matrix_df.index or after_ghost not in prob_matrix_df.columns:
                        return 0.0
                    trans_prob = prob_matrix_df.at[current_state, after_ghost]
                    prob *= trans_prob * ghost_penalty
                    #print(f"Transition {current_state} -> {after_ghost} with ghost state: actual prob {trans_prob} - applying penalty: {trans_prob * ghost_penalty}, cumulative probability: {prob}")
                    i += 2
                else:
                    # due ghost di fila: salta solo uno
                    i += 1
            else:
                break  # fine sequenza

    return prob

def sequence_probability_for_log(group_df: pd.DataFrame, prob_matrix_df: pd.DataFrame, initial_probs_df: pd.DataFrame):
    group_df = group_df.sort_index()

    sequence = group_df['MatchedState'].tolist()

    # Verifica che tutti gli stati siano validi
    for state in sequence:
        if state not in prob_matrix_df.columns:
            print(f"State {state} not found in transition matrix.")
            return np.nan

    # Calcolo probabilità iniziale
    init_prob_map = dict(zip(initial_probs_df['State'], initial_probs_df['Initial_Probability']))
    prob = init_prob_map.get(sequence[0], 0)
    if prob == 0:
        return 0.0
    
    #print(f"Initial Probability for {sequence[0]}: {prob}")

    # Probabilità di transizione
    for i in range(len(sequence) - 1):
        from_state = sequence[i]
        to_state = sequence[i + 1]
        trans_prob = prob_matrix_df.at[from_state, to_state]
        prob *= trans_prob
        if prob == 0:
            break
        #print(f"probability from {from_state} to {to_state}: {trans_prob}, cumulative probability: {prob}")
    
    return prob

def find_closest_adjective(label, target_adj, adj_df):
    if int(label) == -1 or pd.isna(target_adj):
        return (-1, None)

    label = int(label)
    target_adj = float(target_adj)

    candidates = adj_df[adj_df['Label'] == label]

    # Filtra per stesso segno
    same_sign = candidates[candidates['Adjective'] * target_adj >= 0]
    if same_sign.empty:
        return None  # Nessun aggettivo coerente

    # Trova il più vicino per valore assoluto
    closest = same_sign.iloc[(same_sign['Adjective'] - target_adj).abs().argsort()].iloc[0]
    return (label, round(float(closest['Adjective']), 4))

def compute_threshold(df: pd.DataFrame, feature_columns: list, class_stats: dict, quantile=0.05):
    """
    Compute the threshold for classifying points based on their log-likelihood scores.

    Args: 
        df (pd.DataFrame): The input DataFrame containing the data points.
        feature_columns (list): The list of feature columns to consider.
        class_stats (dict): The class statistics containing mean and covariance for each class.
        quantile (float): The quantile to use for thresholding (default is 0.05).
    Returns:
        float: The computed threshold value.
    """
    max_lls = []

    for _, row in df.iterrows():
        x = row[feature_columns].to_numpy()
        lls = []
        for label, params in class_stats.items():
            mu = params["mean"]
            sigma = params["covariance"]
            try:
                ll = multivariate_normal.logpdf(x, mean=mu, cov=sigma, allow_singular=True)
                lls.append(ll)
            except np.linalg.LinAlgError:
                continue
        if lls:
            max_lls.append(max(lls))

    threshold = np.quantile(max_lls, quantile)
    return threshold

def compute_class_stats(df: pd.DataFrame, feature_columns: list, label_column="Label", epsilon=1e-6):
    """
    Compute the mean and covariance for each class in the DataFrame.

    Args: 
        df (pd.DataFrame): The input DataFrame containing the data points.
        feature_columns (list): The list of feature columns to consider.
        label_column (str): The name of the label column (default is "Label").
        epsilon (float): A small value to ensure numerical stability (default is 1e-6).
    Return: 
        dict: A dictionary containing the mean and covariance for each class.
    """
    labels = [l for l in df[label_column].unique() if l != -1]
    stats = {}
    for label in labels:
        class_data = df[df[label_column] == label][feature_columns].to_numpy()
        mu = class_data.mean(axis=0)
        sigma = np.cov(class_data, rowvar=False)
        sigma_reg = sigma + epsilon * np.eye(sigma.shape[0])
        stats[label] = {"mean": mu, "covariance": sigma_reg}
    return stats


def classify_point(x: np.array, class_stats: dict, threshold: float):
    best_label = None
    best_log_likelihood = -np.inf

    for label, params in class_stats.items():
        mu = params["mean"]
        sigma = params["covariance"]
        try:
            ll = multivariate_normal.logpdf(x, mean=mu, cov=sigma, allow_singular=True)
            print(f"Log likelihood for label {label}: {ll}")
        except np.linalg.LinAlgError as e:
            ll = -np.inf
            print(f"Error computing logpdf for label {label}: {e}")

        if ll > best_log_likelihood:
            best_log_likelihood = ll
            best_label = label


    if best_log_likelihood < threshold:
        print("Best label: -1\n")
        return -1  # outlier
    else: 
        print(f"Best label: {best_label}\n")
    return best_label

def segment_user(username:str, file_path: str, features: list, activity: str, scaler: str, threshold: float):
    try: 
        _, rsv, lsv_user = sg.segmentation_on_activity(file_path=file_path, features=features[1:], activity=activity, scaler=scaler, threshold=threshold)
        rsv_df = pd.DataFrame(rsv, columns=['LogNumber'] + [str(i) for i in range(1, len(features) - 1)])
        rsv_df.insert(0, 'Username', username)

        return rsv_df, lsv_user
    except Exception as e:
        print(f"Failed with error: {e}")
        return
    
def compute_user_state(lsv_mean: dict, log_df: pd.DataFrame, adj_df: pd.DataFrame):
    rows = []

    for username, logs in lsv_mean.items():
        for log_key, segs in logs.items():
            log_num = int(log_key.split("_")[1])  # "logNum_2" -> 2
            for seg_key, value in segs.items():
                # costruiamo la riga
                rows.append({
                    "Username": username,  # match con log_df
                    "LogNumber": float(log_num),
                    "Adjective": float(value)
                })

    # dataframe "flat" dal dict
    df_from_dict = pd.DataFrame(rows)

    # merge con log_df rispettando ordine
    df_mean = log_df.copy().reset_index(drop=True)
    df_mean["Adjective"] = df_from_dict["Adjective"].values
    df_mean = df_mean[["Username", "LogNumber", "Adjective", "Label"]]

    df_mean['MatchedState'] = df_mean.apply(
        lambda row: str(find_closest_adjective(row['Label'], row['Adjective'], adj_df)), axis=1
    )

    return df_mean

def main(): 
    username = "grims"
    activity = "ladderActivity"
    recon_activity = "sphereActivity"
    scaler = "standard"
    features = config.ROTATION_FEATURES
    threshold = 0.75

    file_path = config.find_file_from_username(username)
    
    print(f"===== Segmentating user: {username}, activity: {activity}, scaler: {scaler}, threshold: {threshold} =====")
    rsv_df = segment_user(file_path=file_path, features=features, activity=activity, scaler=scaler, threshold=threshold)

    print("\n== RSV DataFrame: ==")
    print(rsv_df)

    print("\n===== Maximum Likelihood Classifier =====")
    eps = 0.25
    min_samples = 8
    file_path = config.PROCESSED_DATA_DIR + f"/dbscan_results_rotation/{recon_activity}/{scaler}" 

    
    # Calcola le statistiche per ogni classe
    df = pd.read_csv(file_path + f"/all_points/dbscan_{recon_activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")
    feature_columns = df.columns[-9:]  # Assuming the last 9 columns are features
    class_stats = compute_class_stats(df, feature_columns)  

    print("\n== Classify New Points ==")
    for index, row in rsv_df.iterrows():
        point = row[-9:].to_numpy()

        label = classify_point(point, class_stats)
        print(f"Point {index} - LogNumber: {row['LogNumber']} - Classified as: {label}")

    log_df = pd.DataFrame({
        'LogNumber': rsv_df['LogNumber'],
        'Adjective': rsv_df['Adjective'],
        'Label': [classify_point(row[-9:].to_numpy(), class_stats) for _, row in rsv_df.iterrows()]
    })

    file_path=config.PROCESSED_DATA_DIR + "/dbscan_results_rotation/sphereActivity/standard"
    adj_df = pd.read_csv(file_path + f"/adjectives/mean/mean_adjectives_{recon_activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")

    # Trova lo stato corrispondente
    log_df['MatchedState'] = log_df.apply(
        lambda row: str(find_closest_adjective(row['Label'], row['Adjective'], adj_df)), axis=1
    )

    #print(log_df)

    prob_matrix_df = pd.read_csv(config.RESULTS_DIR + f"/transition_results/{recon_activity}/{scaler}/transition_probabilities_{recon_activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv", index_col=0)
    initial_probs_df = pd.read_csv(config.RESULTS_DIR + f"/transition_results/{recon_activity}/{scaler}/initial_probabilities_{recon_activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")   
    
    results = []

    for log_number, group in log_df.groupby("LogNumber"):
        print(f"\nProcessing LogNumber: {log_number}..")
        prob = sequence_probability_for_log(group, prob_matrix_df, initial_probs_df)
        results.append((log_number, prob))

    df_sequence_probs = pd.DataFrame(results, columns=["LogNumber", "Sequence_Probability"])

    df_sequence_probs["Normalized"] = (df_sequence_probs["Sequence_Probability"] / df_sequence_probs["Sequence_Probability"].sum())   

    df_sequence_probs["LogProb"] = np.log(df_sequence_probs["Sequence_Probability"].replace(0, np.nan))
    df_sequence_probs["Length"] = log_df.groupby("LogNumber").size().values
    df_sequence_probs["AvgLogProb"] = df_sequence_probs["LogProb"] / df_sequence_probs["Length"]

    print("\n\n== Sequence Probabilities with LogProb and Length: ==")
    print(df_sequence_probs)

    """# WITH GHOST STATE
    ghost_results = []
    for log_number, group in log_df.groupby("LogNumber"):
        print(f"GHOST STATE: Processing LogNumber: {log_number}")
        prob = sequence_probability_ghost_state(group, prob_matrix_df, initial_probs_df)
        ghost_results.append((log_number, prob))

    df_sequence_ghost = pd.DataFrame(ghost_results, columns=["LogNumber", "Sequence_Probability"])

    df_sequence_ghost["Normalized"] = (df_sequence_ghost["Sequence_Probability"] / df_sequence_ghost["Sequence_Probability"].sum())   

    df_sequence_ghost["LogProb"] = np.log(df_sequence_ghost["Sequence_Probability"].replace(0, np.nan))
    df_sequence_ghost["Length"] = log_df.groupby("LogNumber").size().values
    df_sequence_ghost["AvgLogProb"] = df_sequence_ghost["LogProb"] / df_sequence_ghost["Length"]

    print("\n== GHOST STATE: Sequence Probabilities with LogProb and Length: ==")
    print(df_sequence_ghost)"""

    # WITH GHOST STATE PENALTY AT THE END
    print("\n\n== GHOST STATE PENALTY AT THE END ==")
    ghost_end_results = []
    for log_number, group in log_df.groupby("LogNumber"):
        print(f"GHOST STATE: Processing LogNumber: {log_number}")
        prob = sequence_probability_ghost_state_end(group, prob_matrix_df, initial_probs_df)
        ghost_end_results.append((log_number, prob))

    df_sequence_ghost_end = pd.DataFrame(ghost_end_results, columns=["LogNumber", "Sequence_Probability"])

    df_sequence_ghost_end["Normalized"] = (df_sequence_ghost_end["Sequence_Probability"] / df_sequence_ghost_end["Sequence_Probability"].sum())   

    # valori molto negativi sono transizioni molto improbabili 
    df_sequence_ghost_end["LogProb"] = np.log(df_sequence_ghost_end["Sequence_Probability"].replace(0, np.nan))
    df_sequence_ghost_end["Length"] = log_df.groupby("LogNumber").size().values
    df_sequence_ghost_end["AvgLogProb"] = df_sequence_ghost_end["LogProb"] / df_sequence_ghost_end["Length"]

    print("\n== GHOST STATE PENALTY AT THE END: Sequence Probabilities with LogProb and Length: ==")
    print(df_sequence_ghost_end)
    

    return 

if __name__ == "__main__":
    print("== Starting Validation ==\n\n") 
    main()
    exit(1)