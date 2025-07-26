import os 
import numpy as np
import pandas as pd

from data_processing.src import config


def mean_adjectives(filepath: str, dbscan_results: str, activity: str, scaler: str, save_path: str): 
    file_path = config.PROCESSED_DATA_DIR + f"/{dbscan_results}/{activity}/{scaler}"


    # Carica il file
    #adj_df = pd.read_csv(file_path + f"/adjectives/adjectives_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")
    adj_df = pd.read_csv(filepath)

    # Lista per salvare i risultati
    rows = []

    # Raggruppa per Label
    for label, group in adj_df.groupby('Label'):
        adjectives = group['Adjective']
        pos = adjectives[adjectives > 0]
        neg = adjectives[adjectives < 0]

        # Positivi
        if not pos.empty:
            rows.append((label, pos.min()))
            rows.append((label, pos.mean()))
            rows.append((label, pos.max()))
        
        # Negativi
        if not neg.empty:
            rows.append((label, neg.min()))
            rows.append((label, neg.mean()))
            rows.append((label, neg.max()))

    # Crea nuovo DataFrame
    summary_df = pd.DataFrame(rows, columns=["Label", "Adjective"])

    summary_df = summary_df.drop_duplicates(subset=["Label", "Adjective"]).reset_index(drop=True)

    #os.makedirs(file_path + "/adjectives/mean", exist_ok=True)
    # Esporta su CSV
    #summary_df.to_csv(file_path + f"/adjectives/mean/mean_adjectives_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv", index=False)
    summary_df.to_csv(save_path, index=False)

    return summary_df

def percentile_adjectives(filepath: str, dbscan_results: str, activity: str, scaler: str, save_path: str): 
    file_path = config.PROCESSED_DATA_DIR + f"/{dbscan_results}/{activity}/{scaler}"


    # Carica il file
    #adj_df = pd.read_csv(file_path + f"/adjectives/adjectives_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv")
    adj_df = pd.read_csv(filepath)

    # Lista per salvare i risultati
    rows = []

    # Raggruppa per Label
    for label, group in adj_df.groupby('Label'):
        adjectives = group['Adjective']
        pos = adjectives[adjectives > 0]
        neg = adjectives[adjectives < 0]

        # Positivi
        if not pos.empty:
            rows.append((label, pos.quantile(0.25)))
            rows.append((label, pos.quantile(0.75)))
        
        # Negativi
        if not neg.empty:
            rows.append((label, neg.quantile(0.25)))
            rows.append((label, neg.quantile(0.75)))

    # Crea nuovo DataFrame
    summary_df = pd.DataFrame(rows, columns=["Label", "Adjective"])

    summary_df = summary_df.drop_duplicates(subset=["Label", "Adjective"]).reset_index(drop=True)

    #os.makedirs(file_path + "/adjectives/mean", exist_ok=True)
    # Esporta su CSV
    #summary_df.to_csv(file_path + f"/adjectives/mean/mean_adjectives_{activity}_{threshold}_{scaler}_eps{eps}_minsample{min_samples}.csv", index=False)
    summary_df.to_csv(save_path, index=False)

    return summary_df