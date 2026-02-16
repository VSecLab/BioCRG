import os
import pandas as pd
from data_processing.src import config 

path = config.RESULTS_DIR + f"/grid_results/join_grid_results.csv"

def doe(result_df: pd.DataFrame): 
    doe_on_predictions_df = pd.DataFrame() 

    for index, row in result_df.iterrows():
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

    markov_df = pd.read_csv(config.RESULTS_DIR + f"/grid_results/results_markov.csv")
    suffix_df = pd.read_csv(config.RESULTS_DIR + f"/grid_results/results_suffix.csv")

    joined_df = pd.concat([markov_df, suffix_df], axis=0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joined_df.to_csv(path, header=True, index=False)

    print(f"Joined results saved to: {path}\n")
    return joined_df

if __name__ == "__main__":
    #join_results()
    result_df = pd.read_csv(path)
    doe(result_df)
