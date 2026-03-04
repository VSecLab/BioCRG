import os
import itertools
import numpy as np
import pandas as pd
from data_processing.src import config 

path = config.RESULTS_DIR + f"/grid_results/join_grid_results.csv"

def add_ghost_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge la colonna GhostRatio = GhostCount / N per ogni riga.
    Restituisce il dataframe con la nuova colonna.
    """
    df = df.copy()
    df["GhostRatio"] = df.apply(
        lambda row: row["GhostCount"] / row["N"] if row["N"] and row["N"] != 0 else 0,
        axis=1
    )
    return df

def doe(result_df: pd.DataFrame): 
    doe_on_predictions_df = pd.DataFrame() 

    for index, row in result_df.iterrows():
        print(f"Index: {index}")
        target_activity = row["Target_Activity"]
        test_activity = row["TestActivity"]
        threshold = row["Target_Threshold"]
        mode = row["Concat"]
        K = row["K"]
        N = row["N_states"]
        L = row["L"] if row["L"] != "None" else 0
        pmin = row["pmin"] if row["pmin"] != "None" else 0
        gamma_min = row["gamma_min"] if row["gamma_min"] != "None" else 0
        eps_suffix = row["eps_suffix"] if row["eps_suffix"] != "None" else 0
        GhostCount = row["Ghost_Count"]
        GhostRatio = GhostCount / N if N and N != 0 else 0
        Predict = 0 if row["AvgLogProb"] > 10 else 1
        Actual = 1 if target_activity == test_activity else 0
        Matched = Predict == Actual

        doe_on_predictions_df = pd.concat([doe_on_predictions_df, pd.DataFrame([
            {"Target_Activity": target_activity,
            "TestActivity": test_activity,
            "mode": mode,
            "Threshold": threshold,
            "K": K,
            "N": N,
            "GhostCount": GhostCount,
            "GhostRatio": GhostRatio,
            "L": L,
            "pmin": pmin,
            "gamma_min": gamma_min,
            "eps_suffix": eps_suffix,
            "Predict": Predict,
            "Actual": Actual,
            "Matched": Matched}
        ])], ignore_index=True)

    doe_path = config.RESULTS_DIR + f"/recognition_results/doe/doe_on_predictions.csv"
    os.makedirs(os.path.dirname(doe_path), exist_ok=True)

    doe_on_predictions_df.to_csv(doe_path, header=True, mode='a', index=False)

    print(f"DOE on predictions saved to: {doe_path}\n")
    print("DOE on predictions: \n")
    print(doe_on_predictions_df)

def join_results(): 

    markov_df = pd.read_csv(config.RESULTS_DIR + f"/grid_results/results_markov_mp.csv")
    suffix_df = pd.read_csv(config.RESULTS_DIR + f"/grid_results/results_suffix_mp.csv")

    # Prendi le colonne di suffix_df come riferimento
    suffix_columns = suffix_df.columns.tolist()
    
    # Riorganizza markov_df per avere le stesse colonne di suffix_df nello stesso ordine
    # Aggiungi le colonne mancanti con valore None
    for col in suffix_columns:
        if col not in markov_df.columns:
            markov_df[col] = None
    
    # Riordina le colonne di markov_df per matchare quelle di suffix_df
    markov_df = markov_df[suffix_columns]

    # Concatena i due dataframe
    joined_df = pd.concat([markov_df, suffix_df], axis=0, ignore_index=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joined_df.to_csv(path, header=True, index=False)

    print(f"Joined results saved to: {path}\n")
    return joined_df

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tutte le possibili combinazioni dei valori specificati e,
    per ognuna, conta il numero di righe nel dataframe che la rispettano.
    Le combinazioni assenti nel dataframe avranno Count = 0.
    """
    group_cols = ["Target_Activity", "TestActivity", "mode", "GhostRatio", "Actual"]

    domain = {
        "Target_Activity": ["sphereActivity", "ladderActivity"],
        "TestActivity":    ["sphereActivity", "ladderActivity"],
        "mode":            ["markov", "suffix_tree"],
        #"Threshold":       [0.5, 0.6, 0.7],
        #"GhostCount":      list(range(0, 11)),
        "GhostRatio":      [round(i * 0.1, 1) for i in range(11)],
        "Actual":          [0, 1],
    }

    all_combinations = pd.DataFrame(
        list(itertools.product(*domain.values())),
        columns=domain.keys()
    )

    counts = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="Count")
    )

    result = all_combinations.merge(counts, on=group_cols, how="left").fillna({"Count": 0})
    result["Count"] = result["Count"].astype(int)

    return result

def filtering(save = True):
    doe_df_path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe/doe_on_predictions.csv"

    doe_df = pd.read_csv(doe_df_path)
    doe_df = add_ghost_ratio(doe_df)

    df_markov = doe_df[doe_df["mode"] == "markov"].copy()
    df_suffix = doe_df[doe_df["mode"] == "suffix_tree"]
    df_suffix = df_suffix[
        (df_suffix["L"] == 4) &
        (df_suffix["pmin"] == 0.000001) &
        (df_suffix["gamma_min"] == 0.0) &
        (df_suffix["eps_suffix"] == 0.0)
    ].copy()

    for _df in [df_markov, df_suffix]:
        _df["GhostRatio"] = (np.floor(_df["GhostRatio"] * 10) / 10).round(1)

    counts_df_markov = filter_data(df_markov)
    counts_df_suffix = filter_data(df_suffix)

    counts_df = pd.concat([counts_df_markov, counts_df_suffix], axis=0, ignore_index=True)

    path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe"

    if save: 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        counts_df.to_csv(path + "/filtered_counts.csv", header=True, index=False, mode='w')

    suffix_df = counts_df[counts_df["mode"] == "suffix_tree"]
    markov_df = counts_df[counts_df["mode"] == "markov"]
    
    if save: 
        suffix_df.to_csv(path + "/filtered_counts_suffix.csv", header=True, index=False, mode='w')
        markov_df.to_csv(path + "/filtered_counts_markov.csv", header=True, index=False, mode='w')

    return markov_df, suffix_df

def sample_by_min_count(
    min_count: int = 10,
    counts_csv_path: str = None,
    doe_csv_path: str = None,
    output_path: str = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Selects all combinations from the counts CSV whose Count >= min_count,
    then for each combination randomly samples exactly `min_count` rows from
    doe_on_predictions.csv (taking all rows when Count == min_count).
    All sampled rows are concatenated and saved to output_path.

    Parameters
    ----------
    min_count : int
        Minimum count threshold. Only combinations with Count >= min_count
        are considered; exactly min_count rows are sampled per combination.
    counts_csv_path : str, optional
        Path to the filtered counts CSV (e.g. filtered_counts_markov.csv).
        Defaults to the standard doe results directory.
    doe_csv_path : str, optional
        Path to doe_on_predictions.csv.
    output_path : str, optional
        Where to save the resulting CSV. Defaults to
        sampled_min<min_count>.csv in the doe results directory.
    random_state : int
        Seed for reproducible random sampling.

    Returns
    -------
    pd.DataFrame
        DataFrame with the sampled rows.
    """
    base_path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe"

    if counts_csv_path is None:
        counts_csv_path = base_path + "/filtered_counts_markov.csv"
    if doe_csv_path is None:
        doe_csv_path = base_path + "/doe_on_predictions.csv"
    if output_path is None:
        output_path = base_path + f"/sampled_min{min_count}.csv"

    group_cols = ["Target_Activity", "TestActivity", "mode", "Actual", "GhostRatio"]

    counts_df = pd.read_csv(counts_csv_path)
    doe_df = pd.read_csv(doe_csv_path)

    # Keep only combinations that meet the threshold
    eligible = counts_df[counts_df["Count"] >= min_count]

    if eligible.empty:
        print(f"No combinations found with Count >= {min_count}.")
        return pd.DataFrame()

    sampled_rows = []
    for _, combo_row in eligible.iterrows():
        # Build a boolean mask matching all group columns
        mask = pd.Series([True] * len(doe_df), index=doe_df.index)
        for col in group_cols:
            mask &= doe_df[col] == combo_row[col]

        matching = doe_df[mask]

        if len(matching) < min_count:
            print(f"Warning: combination has {len(matching)} rows in doe_on_predictions "
                  f"but Count column says {int(combo_row['Count'])}. Skipping.")
            continue

        sampled = matching.sample(n=min_count, random_state=random_state)
        sampled_rows.append(sampled)

    if not sampled_rows:
        print("No rows were sampled.")
        return pd.DataFrame()

    result = pd.concat(sampled_rows, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, header=True, index=False, mode='w')

    print(f"Sampled dataset saved to: {output_path}")
    print(f"  Eligible combinations : {len(eligible)}")
    print(f"  Total sampled rows    : {len(result)}  ({min_count} per combination)")

    return result


if __name__ == "__main__":
    #rename_mode()

    #join_results()
    #result_df = pd.read_csv(path)
    #doe(result_df)

    #markov_df, suffix_df = filtering(save = False)
    #mode = "markov"
    #count_path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe/filtered_counts_markov.csv"
    #doe_csv_path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe/doe_on_predictions.csv"
    #path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe/min_count"
    
    filtering()
    
    mode = "suffix"
    count_path = f"/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe/filtered_counts_{mode}.csv"
    doe_csv_path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe/doe_on_predictions.csv"
    path = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/doe/min_count"
    
    os.makedirs(path, exist_ok=True)
    output_path = path + f"/{mode}_sampled_min10.csv"
    res = sample_by_min_count(min_count=10, counts_csv_path=count_path, doe_csv_path=doe_csv_path, output_path=output_path)
    

    
