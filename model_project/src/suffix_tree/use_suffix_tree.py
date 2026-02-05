import pandas as pd
from suffix_tree import create_tree, PST_Probability

processed_data_path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs" +  "/processed" 
result_directory = "kmeans_results"
target_activity = "sphereActivity"
scaler = "standard"
file_path = processed_data_path + f"/{result_directory}/{target_activity}/{scaler}"

def find_closest_adjective(adj_df: pd.DataFrame, label, target_adj):
    candidates = adj_df[adj_df['Label'] == label]
    same_sign = candidates[candidates['Adjective'] * target_adj >= 0]
    if same_sign.empty:
        return None  # Nessun aggettivo con stesso segno
    closest = same_sign.iloc[(same_sign['Adjective'] - target_adj).abs().argsort()].iloc[0]
    return (label, closest['Adjective'])

def extract_activity_sequences(file_path: str, activity: str, scaler: str, threshold: float, k: int):
    """
    Estrae le sequenze di azioni per ogni singola attività svolta dagli utenti.
    
    Args:
        file_path: Percorso base dei file
        activity: Nome dell'attività (es. 'sphereActivity')
        scaler: Tipo di scaler utilizzato (es. 'standard')
        threshold: Soglia utilizzata
        k: Numero di cluster
    
    Returns:
        list: Lista di liste, dove ogni lista interna rappresenta una singola attività
              e contiene tuple (adjective, label) per ogni azione
    """
    # Carica i dati
    adj_df = pd.read_csv(file_path + f"/adjectives/adjectives_{activity}_{threshold}_{scaler}_k{k}.csv")
    activity_df = pd.read_csv(file_path + f"/all_points/kmeans_{activity}_{threshold}_{scaler}_k{k}.csv")
    
    # Raggruppa per Username e LogNumber (ogni gruppo è una singola attività)
    grouped = activity_df.groupby(['Username', 'LogNumber'])
    
    sequences = []
    
    # Per ogni attività
    for (username, log_number), group in grouped:
        group_sorted = group.sort_index()  # Ordina per indice temporale
        sequence = []
        
        for _, row in group_sorted.iterrows():
            label = int(row['Label'])
            adjective = float(row['Adjective'])
            
            if label == -1:
                # Stato speciale: usa None per l'aggettivo
                matched_adjective = None
            else:
                # Trova l'aggettivo più vicino usando la logica esistente
                matched_state = find_closest_adjective(adj_df, label, adjective)
                if matched_state:
                    matched_adjective = matched_state[1]  # Estrae solo l'aggettivo
                else:
                    matched_adjective = None  # Se non trova match
            
            # Aggiungi la coppia (adjective, substantive/label) alla sequenza
            sequence.append((matched_adjective, label))

        #print(f"[{username} - {log_number}]:\n{sequence}\n\n")
        sequences.append(sequence)
    
    return sequences

def assign_symbolic_states(sequences):
    """
    Assegna un valore simbolico a ogni stato unico trovato nelle sequenze.
    
    Args:
        sequences: Lista di sequenze, dove ogni sequenza contiene tuple (adjective, label)
    
    Returns:
        tuple: (state_mapping, symbolic_sequences)
            - state_mapping: dizionario {stato_originale: identificatore_simbolico}
            - symbolic_sequences: lista di sequenze con stati simbolici
    """
    # Estrai tutti gli stati unici
    unique_states = set()
    for sequence in sequences:
        for state in sequence:
            unique_states.add(state)
    
    # Ordina gli stati per coerenza (prima per label, poi per adjective)
    # None viene trattato separatamente
    sorted_states = sorted(unique_states, key=lambda x: (x[1] if x[1] is not None else -2, 
                                                          x[0] if x[0] is not None else float('-inf')))
    
    # Crea il mapping: assegna "c0", "c1", "c2", etc.
    state_mapping = {}
    for idx, state in enumerate(sorted_states):
        state_mapping[state] = f"c{idx}"
    
    # Converti tutte le sequenze usando il mapping
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
                       dove ogni lista contiene tuple (adjective, label)
    
    Returns:
        dict: dizionario con la stessa struttura ma con le sequenze convertite in simboli
    """
    converted_sequences = {}
    
    for username, log_dict in user_sequences.items():
        converted_sequences[username] = {}
        
        for log_number, sequence in log_dict.items():
            # Converti ogni elemento della sequenza usando state_mapping
            converted_seq = [state_mapping[state] for state in sequence]
            converted_sequences[username][log_number] = converted_seq
    
    return converted_sequences

def compute_sequences(file_path:str, target_activity:str, scaler:str, threshold:float=0.6, k:int=10): 
    sequences = extract_activity_sequences(
        file_path=file_path, 
        activity=target_activity, 
        scaler=scaler, 
        threshold=threshold, 
        k=k
    )

    #print(f"\n {sequences}")

    # Usa la funzione
    state_mapping, symbolic_sequences = assign_symbolic_states(sequences)
    return symbolic_sequences, state_mapping

def main():
    symbolic_sequences, state_mapping = compute_sequences(
        file_path=file_path,
        target_activity=target_activity,
        scaler=scaler,
        threshold=0.6,
        k=10
    )
    L = 5

    print(f"state_mapping: {state_mapping}")
    print(f"alphabet: {list(state_mapping.values())}")
    # Create the suffix tree
    print("\n" + "=" * 60)
    print("Creating Suffix Tree...")
    print("=" * 60)
    suffix_tree = create_tree(sequences=symbolic_sequences, alphabet=list(state_mapping.values()), L=L)

    print("\n" + "=" * 60)
    print(f"Computing probability for test sequence: {symbolic_sequences}")
    print("=" * 60)
    for i in range(0,4):
        PST_Probability(suffix_tree, symbolic_sequences[i], L=L)



