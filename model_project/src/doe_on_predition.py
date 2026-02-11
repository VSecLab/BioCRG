import pandas as pd
import os
from data_processing.src import config 
path = config.RESULTS_DIR + f"/recognition_results/results_gen.csv"
result_df = pd.read_csv(config.RESULTS_DIR + f"/recognition_results/results_gen.csv")

def doe(): 
    doe_on_predictions_df = pd.DataFrame() 

    columns = ["mode", "N", "GhostCount", "L", "Predict", "Actual", "Matched"]

    for index, row in result_df.iterrows():
        target_activity = row["Target_Activity"]
        test_activity = row["TestActivity"]
        mode = row["Concat"]
        N = row["N_states"]
        L = row["L"] if row["L"] != "None" else 0
        GhostCount = row["Ghost_Count"]
        Predict = 0 if row["AvgLogProb"] > 10 else 1
        Actual = 1 if target_activity == test_activity else 0
        Matched = Predict == Actual

        doe_on_predictions_df = pd.concat([doe_on_predictions_df, pd.DataFrame([
            {"mode": mode,
            "N": N,
            "GhostCount": GhostCount,
            "L": L,
            "Predict": Predict,
            "Actual": Actual,
            "Matched": Matched}
        ])], ignore_index=True)

    doe_path = config.RESULTS_DIR + f"/recognition_results/doe_on_predictions.csv"
    os.makedirs(os.path.dirname(doe_path), exist_ok=True)

    doe_on_predictions_df.to_csv(doe_path, header=True, mode='a', index=False)

    print(f"DOE on predictions saved to: {doe_path}\n")
    print("DOE on predictions: \n")
    print(doe_on_predictions_df)

def main(): 
    result_df["L"] = "None"
    for index, row in result_df.iterrows():
        row["L"] = 5 if row["Concat"] == "suffix_tree" else "None"
        result_df.at[index, "L"] = row["L"]

    print(result_df)
    result_df.to_csv(path, header=True, index=False)


if __name__ == "__main__":
    doe()
