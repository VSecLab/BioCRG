import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from data_processing.src import config

def compute_transition_probabilities(df_transition: pd.DataFrame):
    row_sums = df_transition.sum(axis=1)

    df_prob = df_transition.div(row_sums, axis=0).fillna(0)

    return df_prob

def heatmap_plot(df_transition: pd.DataFrame):
    plt.figure(figsize=(16, 10))
    sns.heatmap(df_transition, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=True, yticklabels=True)
    plt.title("Heatmap of Transition Matrix")
    plt.xlabel("Next State")
    plt.ylabel("Actual State")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()

    return 

def heatmap_plot_probabilities(df_probabilities: pd.DataFrame):
    plt.figure(figsize=(16, 10))
    sns.heatmap(df_probabilities, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=True, yticklabels=True)
    plt.title("Probability Heatmap of Transition Matrix")
    plt.xlabel("Next State")
    plt.ylabel("Actual State")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()

def find_closest_adjective(adj_df: pd.DataFrame, label, target_adj):
    candidates = adj_df[adj_df['Label'] == label]
    same_sign = candidates[candidates['Adjective'] * target_adj >= 0]
    if same_sign.empty:
        return None  # Nessun aggettivo con stesso segno
    closest = same_sign.iloc[(same_sign['Adjective'] - target_adj).abs().argsort()].iloc[0]
    return (label, closest['Adjective'])

def format_state(state):
    label, adj = state
    if label == -1:
        return "(-1, None)"
    else:
        return f"({label}, {round(adj, 4)})"

def transition_matrix_computation(adj_df: pd.DataFrame, activity_df: pd.DataFrame):
    # lista ordinata di tutti gli stati unici come (label, adjective)
    state_list = adj_df.copy() 
    
    state_list['State'] = list(zip(state_list['Label'], state_list['Adjective']))
    state_list = state_list['State'].tolist()
    state_list.append((-1, None))

    # mappa per ottenere l'indice di uno stato nella matrice
    state_index_map = {state: idx for idx, state in enumerate(state_list)}

    # matrice di transizione
    n_states = len(state_list)
    transition_matrix = np.zeros((n_states, n_states), dtype=int)

    grouped = activity_df.groupby(['Username', 'LogNumber'])

    initial_states = []
    sequence_lengths = []  # Lista per memorizzare la lunghezza di ogni sequenza

    # per ogni gruppo
    for _, group in grouped:
        group_sorted = group.sort_index()  # ordina per indice
        states = []

        first_row = group.sort_index().iloc[0]
        first_label = int(first_row['Label'])
        first_adjective = float(first_row['Adjective'])
           
        if first_label == -1:
            matched_state = (-1, None)
        else:
            matched_state = find_closest_adjective(adj_df, first_label, first_adjective)
        if matched_state in state_index_map:
            initial_states.append(matched_state)

        for _, row in group_sorted.iterrows():
            label = int(row['Label'])  
            adjective = float(row['Adjective'])

            if label == -1:
                # uso lo stato extra senza cercare aggettivo
                matched_state = (-1, None)
            else:
                matched_state = find_closest_adjective(adj_df, label, adjective)

            if matched_state and matched_state in state_index_map:
                states.append(matched_state)

        # Memorizza la lunghezza della sequenza
        sequence_lengths.append(len(states))

        # conta le transizioni
        for i in range(len(states) - 1):
            from_idx = state_index_map[states[i]]
            to_idx = state_index_map[states[i+1]]
            transition_matrix[from_idx, to_idx] += 1

    counter = Counter(initial_states)
    total = sum(counter.values())

    # Calcola il numero medio di stati per sequenza
    avg_states_per_sequence = np.mean(sequence_lengths) if sequence_lengths else 0
    #print(f"Average: {avg_states_per_sequence:.2f}")
    #print(f"Total sequences: {len(sequence_lengths)}")
    #print(f"Minimum sequence length: {min(sequence_lengths) if sequence_lengths else 0}")
    #print(f"Maximum sequence length: {max(sequence_lengths) if sequence_lengths else 0}")

    # Probabilità iniziali
    initial_prob_vector = np.array([counter.get(state, 0) / total for state in state_list])

    # Crea DataFrame con etichette leggibili
    df_initial = pd.DataFrame({
        'State': [format_state(s) for s in state_list],
        'Initial_Probability': initial_prob_vector
    })

    state_labels = [format_state(s) for s in state_list]

    df_transition = pd.DataFrame(transition_matrix, index=state_labels, columns=state_labels)

    df_probabilities = compute_transition_probabilities(df_transition=df_transition)

    # Crea un dizionario con le statistiche delle sequenze
    sequence_stats = {
        'avg_states_per_sequence': avg_states_per_sequence,
        'total_sequences': len(sequence_lengths),
        'min_sequence_length': min(sequence_lengths) if sequence_lengths else 0,
        'max_sequence_length': max(sequence_lengths) if sequence_lengths else 0,
        'sequence_lengths': sequence_lengths
    }

    return df_transition, df_probabilities, df_initial, sequence_stats
    
def transition_computation_dbscan(activity: str, scaler: str, threshold: float, eps: float, min_samples: int, file_path: str, enable_plots: bool = False):
    
    #sub_df = pd.read_csv(file_path + f"/substantives/substantives_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")
    
    adj_df = pd.read_csv(file_path + f"/adjectives/adjectives_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")
    activity_df = pd.read_csv(file_path + f"/all_points/dbscan_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")

    df_transition, df_probabilities, df_initial, sequence_stats = transition_matrix_computation(adj_df, activity_df)

    if enable_plots:
        print("\n== Transition Matrix: ==")
        #print(df_transition)
        heatmap_plot(df_transition)

        print("\n== Transition Probabilities: ==")
        #print(df_probabilities)
        heatmap_plot_probabilities(df_probabilities)

    """print("\n== Initial State Probabilities: ==")
    print(df_initial)"""

    os.makedirs(config.RESULTS_DIR + f"/transition_results/{activity}/{scaler}", exist_ok=True)
    df_transition.to_csv(config.RESULTS_DIR + f"/transition_results/{activity}/{scaler}/transition_matrix_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")
    df_probabilities.to_csv(config.RESULTS_DIR + f"/transition_results/{activity}/{scaler}/transition_probabilities_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")
    df_initial.to_csv(config.RESULTS_DIR + f"/transition_results/{activity}/{scaler}/initial_probabilities_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv", index=False)    
    print(f"\nTransition results saved in {config.RESULTS_DIR}/transition_results/{activity}/{scaler}")

    return df_transition, df_probabilities, df_initial, sequence_stats

def transition_computation_kmeans(activity: str, scaler: str, threshold: float, k: int, file_path: str, enable_plots: bool = False):
    
    #sub_df = pd.read_csv(file_path + f"/substantives/substantives_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")
    
    adj_df = pd.read_csv(file_path + f"/adjectives/adjectives_{activity}_{threshold}_{scaler}_k{k}.csv")
    activity_df = pd.read_csv(file_path + f"/all_points/kmeans_{activity}_{threshold}_{scaler}_k{k}.csv")

    df_transition, df_probabilities, df_initial, sequence_stats = transition_matrix_computation(adj_df, activity_df)

    if enable_plots:
        print("\n== Transition Matrix: ==")
        #print(df_transition)
        heatmap_plot(df_transition)

        print("\n== Transition Probabilities: ==")
        #print(df_probabilities)
        heatmap_plot_probabilities(df_probabilities)

    """print("\n== Initial State Probabilities: ==")
    print(df_initial)"""

    os.makedirs(config.RESULTS_DIR + f"/transition_results/{activity}/{scaler}", exist_ok=True)
    df_transition.to_csv(config.RESULTS_DIR + f"/transition_results/{activity}/{scaler}/transition_matrix_{activity}_{threshold}_{scaler}_k{k}.csv")
    df_probabilities.to_csv(config.RESULTS_DIR + f"/transition_results/{activity}/{scaler}/transition_probabilities_{activity}_{threshold}_{scaler}_k{k}.csv")
    df_initial.to_csv(config.RESULTS_DIR + f"/transition_results/{activity}/{scaler}/initial_probabilities_{activity}_{threshold}_{scaler}_k{k}.csv", index=False)    
    print(f"\nTransition results saved in {config.RESULTS_DIR}/transition_results/{activity}/{scaler}")

    return df_transition, df_probabilities, df_initial, sequence_stats

