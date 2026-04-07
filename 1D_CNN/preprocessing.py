import os
import json
import joblib
import numpy as np
import pandas as pd

from typing import List
from pathlib import Path
from sklearn.preprocessing import StandardScaler


MODULE_DIR = Path(__file__).resolve().parent

with open(MODULE_DIR / "config.json", "r") as f:
    config = json.load(f)

TIMESTEPS = config["timesteps"]
STRIDE = config["stride"]
METADATA_COLUMNS = {"Activity", "LogNumber", "Timestamp", "Index"}


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric sensor columns used as CNN inputs."""

    feature_df = df.drop(columns=[col for col in METADATA_COLUMNS if col in df.columns], errors="ignore")
    return feature_df.select_dtypes(include=[np.number]).dropna()


def _group_columns(df: pd.DataFrame) -> List[str]:
    """Columns that define an independent temporal segment."""

    return [col for col in ["LogNumber"] if col in df.columns]


def _sort_columns(df: pd.DataFrame) -> List[str]:
    """Prefer timestamp order, then row order when available."""

    return [col for col in ["Timestamp", "Index"] if col in df.columns]

def fit_and_save_scaler(X_train, path="outputs/scalers/activity.pkl"):
    """
    X_train: (N_windows, 50, 18) — finestre utenti train, valori grezzi
    """
    N, T, F = X_train.shape
    # (N*50, 18): ogni timestep e' un sample per il fit
    scaler = StandardScaler().fit(X_train.reshape(-1, F))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler salvato: {path}")
    return scaler


def apply_scaler(X, scaler):
    """Mantiene shape (N, T, F). Restituisce float32 per TF."""
    N, T, F = X.shape
    return scaler.transform(
        X.reshape(-1, F)
    ).reshape(N, T, F).astype("float32")


def load_scaler(path):
    return joblib.load(path)

def load_data(file_path: str) -> pd.DataFrame: 
    """
    Load data from a CSV file into a pandas DataFrame.
    Automatically handles non-numeric values by converting them to NaN and ensuring correct data types.
    
    Args:
        file_path (str) : The path to the CSV file.
        
    Returns:
        pd.DataFrame    : The loaded data as a DataFrame with correct numeric types.
    """

    try:
        df = pd.read_csv(file_path, index_col=None, low_memory=False)
        
        # le colonne numeriche sono tutte quelle che non sono 'Activity'
        numeric_columns = [col for col in df.columns if col not in ['Activity']]
        
        # converto le colonne numeriche in tipo float, gestendo gli errori
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        #print(f"Data successfully loaded - Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  


def sliding_window(data: np.ndarray,
                   timesteps: int,
                   stride: int) -> np.ndarray:
    """
    Trasforma una serie temporale in finestre sovrapposte.

    Args:
        data      : array (T, F) — T timestep, F features
        timesteps : ampiezza finestra
        stride    : passo di scorrimento

    Returns:
        windows   : array (N, timesteps, F)
                    N = floor((T - timesteps) / stride) + 1
    """
    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    T, F = data.shape
    if T < timesteps:
        return np.empty((0, timesteps, F), dtype=np.float32)

    n_windows = (T - timesteps) // stride + 1

    windows = np.empty((n_windows, timesteps, F), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        windows[i] = data[start : start + timesteps]

    return windows

def load_session(folder_path: str, activity: str) -> np.ndarray:
    """
    Legge il CSV presente nella cartella e restituisce le finestre per
    ogni segmento indipendente della activity richiesta.

    Returns:
        array (N_windows, TIMESTEPS, N_FEATURES)
    """
    # Trova il CSV (assumiamo uno solo)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if len(csv_files) == 0:
        raise ValueError(f"Nessun CSV trovato in {folder_path}")
    if len(csv_files) > 1:
        raise ValueError(f"Più CSV trovati in {folder_path}, non atteso")

    fname = csv_files[0]
    path  = os.path.join(folder_path, fname)

    df   = load_data(path)
    feature_count = _feature_frame(df).shape[1]
    filtered_df = filter_data_on_activity(df, activity=activity)
    if filtered_df.empty:
        return np.empty((0, TIMESTEPS, feature_count), dtype=np.float32)

    segment_columns = _group_columns(filtered_df)
    sort_columns = _sort_columns(filtered_df)
    feature_count = _feature_frame(filtered_df).shape[1]

    all_windows = []

    if segment_columns:
        grouped_segments = filtered_df.groupby(segment_columns, sort=False, dropna=False)
    else:
        grouped_segments = [(None, filtered_df)]

    for _, segment_df in grouped_segments:
        ordered_segment = segment_df.sort_values(sort_columns, kind="mergesort") if sort_columns else segment_df
        features = _feature_frame(ordered_segment)
        if features.empty:
            continue

        wins = sliding_window(features.to_numpy(dtype=np.float32), TIMESTEPS, STRIDE)
        if wins.size == 0:
            continue

        all_windows.append(wins)

    if len(all_windows) == 0:
        return np.empty((0, TIMESTEPS, feature_count), dtype=np.float32)

    wins = np.concatenate(all_windows, axis=0)

    print(f"{os.path.basename(folder_path)}/{fname}: "
            f"{filtered_df.shape[0]} righe filtrate -> {wins.shape[0]} finestre")

    return wins


def load_session_executions(folder_path: str, activity: str):
    """
    Restituisce una lista di esecuzioni indipendenti per una sessione utente.
    Ogni elemento e': (execution_id, windows) dove windows ha shape
    (N_windows_exec, TIMESTEPS, N_FEATURES).
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if len(csv_files) == 0:
        raise ValueError(f"Nessun CSV trovato in {folder_path}")
    if len(csv_files) > 1:
        raise ValueError(f"Più CSV trovati in {folder_path}, non atteso")

    fname = csv_files[0]
    path = os.path.join(folder_path, fname)

    df = load_data(path)
    filtered_df = filter_data_on_activity(df, activity=activity)
    if filtered_df.empty:
        return []

    segment_columns = _group_columns(filtered_df)
    sort_columns = _sort_columns(filtered_df)

    if segment_columns:
        grouped_segments = filtered_df.groupby(segment_columns, sort=False, dropna=False)
    else:
        grouped_segments = [("all", filtered_df)]

    executions = []

    for seg_key, segment_df in grouped_segments:
        ordered_segment = segment_df.sort_values(sort_columns, kind="mergesort") if sort_columns else segment_df
        features = _feature_frame(ordered_segment)
        if features.empty:
            continue

        wins = sliding_window(features.to_numpy(dtype=np.float32), TIMESTEPS, STRIDE)
        if wins.size == 0:
            continue

        if isinstance(seg_key, tuple):
            seg_name = "_".join(str(x) for x in seg_key)
        else:
            seg_name = str(seg_key)

        execution_id = f"{os.path.basename(folder_path)}/{fname}:{seg_name}"
        executions.append((execution_id, wins))

    return executions

def filter_data_on_activity(df: pd.DataFrame, activity: str):
    """
    Filter the DataFrame to include only rows with a specific activity.
    
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        activity (str): The activity to filter by.
        
    Returns:
        pd.DataFrame: The filtered DataFrame containing only the specified activity.
    """
    
    try:
        filtered_df = df[df['Activity'] == activity]
        return filtered_df
    except KeyError as e:
        print(f"Error filtering data on activity: {e}")
        return pd.DataFrame()

def load_all_sessions(csv_dir: str, activity: str) -> np.ndarray:
    """
    Legge tutte le sotto-cartelle (tranne 'test_data'),
    aggrega le finestre e restituisce un unico array.

    Returns:
        X : array (N_total, TIMESTEPS, N_FEATURES)
    """
    all_windows = []

    for subfolder in sorted(os.listdir(csv_dir)):
        subfolder_path = os.path.join(csv_dir, subfolder)

        # Skip se non è directory o è test_data
        if not os.path.isdir(subfolder_path) or subfolder == "test_data":
            continue

        wins = load_session(subfolder_path, activity=activity)
        if wins.shape[0] == 0:
            continue
        all_windows.append(wins)

    if len(all_windows) == 0:
        raise ValueError("Nessun dato trovato!")

    X = np.concatenate(all_windows, axis=0)

    print(f"\nDataset totale: {X.shape}  (finestre, timesteps, features)")
    return X


if __name__ == "__main__":
    test_dir = config["raw_data"]
    activity = "sphereActivity"

    X = load_all_sessions(csv_dir=test_dir, activity=activity)
    # Esempio singola finestra → quello che vedrà la CNN
    print("Shape singola finestra:", X[0].shape)   # (50, 21)


