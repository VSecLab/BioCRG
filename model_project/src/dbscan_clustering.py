import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from data_processing.src import etl, config, plot
import segmentation as sg 

def get_substantives_dataframe(df: pd.DataFrame, unique_labels: np.ndarray):
    """
    Create a DataFrame with cluster labels and their corresponding centers.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to cluster.
        unique_labels (np.ndarray): Array of unique cluster labels.
    Returns:
        pd.DataFrame: A DataFrame where each row represents a cluster with its label and center coordinates.
    """
    
    # Calculate centers for each cluster
    centers_df = df.groupby('Label').mean()
    
    # Filter only the labels we want
    centers_df = centers_df.loc[unique_labels]
    
    # Reset index to make 'Label' a column
    centers_df = centers_df.reset_index()
    
    return centers_df

def get_adjectives_dataframe(df: pd.DataFrame, unique_labels: np.ndarray):
    """
    Create a DataFrame mapping cluster labels to adjectives.
    
    Args:
        df (pd.DataFrame): The original DataFrame containing the 'Adjective' column.
        unique_labels (np.ndarray): The array of cluster labels.
        
    Returns:
        pd.DataFrame: A DataFrame with columns 'Label' and 'Adjective' where each row represents 
                     a label-adjective pair.
    """
    
    # Create lists to store label-adjective pairs
    labels_list = []
    adjectives_list = []
    
    for label in unique_labels:
        adjectives = list(df.loc[df['Label'] == label, 'Adjective'])
        for adjective in adjectives:
            labels_list.append(label)
            adjectives_list.append(adjective)
    
    # Create DataFrame
    adjectives_df = pd.DataFrame({
        'Label': labels_list,
        'Adjective': adjectives_list
    })
    
    return adjectives_df

def grid(df: pd.DataFrame): 

    eps_to_test = [round(eps,2) for eps in np.arange(0.1, 0.55, 0.05)]
    min_samples_to_test = range(3, 16, 1)

    # Dataframe per la metrica sul numero di cluster
    results_clusters = pd.DataFrame( 
        data = np.zeros((len(eps_to_test),len(min_samples_to_test))), # Empty dataframe
        columns = min_samples_to_test, 
        index = eps_to_test
    )

    results_outliers = pd.DataFrame(
        data = np.zeros((len(eps_to_test),len(min_samples_to_test))), # Empty dataframe
        columns = min_samples_to_test, 
        index = eps_to_test
    )

    i = 0

    print(" ITER| INFO%s | CLUS    OUT   |   Cluster Size" % (" "*39))
    print("-"*84)

    for eps in eps_to_test:
        for min_samples in min_samples_to_test:
            
            i += 1
            
            # Calcolo le metriche
            cluster_metric, outliers_metric = scanning(df, eps, min_samples, i)
            
            # Inserisco i risultati nei relativi dataframe
            results_clusters.loc[eps, min_samples] = cluster_metric
            results_outliers.loc[eps, min_samples] = outliers_metric


    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(18,8))

    print("\n== Results of the grid search ==\n")
    sns.heatmap(results_clusters, annot = True, ax = ax1, cbar = False, cmap='coolwarm').set_title("METRIC: Number of clusters")
    sns.heatmap(results_outliers, annot = True, ax = ax2, cbar = False, cmap='coolwarm').set_title("METRIC: Number of outliers")

    ax1.set_xlabel("N"); ax2.set_xlabel("N")
    ax1.set_ylabel("EPSILON"); ax2.set_ylabel("EPSILON")

    plt.tight_layout(); plt.show(); plt.close()

    return

def scanning(df: pd.DataFrame, eps: float, min_samples: int, iter: int):

    dbscan_model_ = DBSCAN(eps = eps, min_samples = min_samples, metric='correlation')
    dbscan_model_.fit(df)
        
    # Number of found Clusters metric 
    
    number_of_clusters = len(set(dbscan_model_.labels_[dbscan_model_.labels_ >= 0]))
    number_of_outliers = len(dbscan_model_.labels_[dbscan_model_.labels_ == -1])

    # Calculate points per cluster
    unique_labels = set(dbscan_model_.labels_)
    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:  # Exclude noise points
            cluster_sizes[label] = len(dbscan_model_.labels_[dbscan_model_.labels_ == label])

    if cluster_sizes and number_of_clusters > 1:
        cluster_info = ", ".join([f"C{label}: {size}" for label, size in sorted(cluster_sizes.items())])
        #print("    | Cluster sizes: %s" % cluster_info)
    else: 
        cluster_info = "None"
    
    print(" %3d | Tested with eps = %3s and min_samples = %3s | %3s %7s   | %5s" % (iter, eps, min_samples, number_of_clusters, number_of_outliers, cluster_info))
    
        
    return(number_of_clusters, number_of_outliers) 
    
def dbscan_clustering(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5):
    """
    Perform DBSCAN clustering on the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to cluster.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
        np.ndarray: An array of cluster labels, where -1 indicates noise.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='correlation').fit(df)
    

    return db.labels_ 

def box_plot(df: pd.DataFrame):
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df, orient="h")
    plt.title("Box Plot of Features")
    plt.xlabel("Values")
    plt.ylabel("Features")
    plt.show()
    plt.close()

def pair_plot(df: pd.DataFrame):
    plt.figure(figsize=(15, 6))
    sns.pairplot(df)
    plt.show()
    plt.close()

def pair_plot_cluster(df: pd.DataFrame):
    plt.figure(figsize=(15, 6))
    sns.pairplot(df, hue="Label", palette="tab10", diag_kind="hist")
    plt.suptitle("Pairplot PCA con cluster DBSCAN", y=1.02)
    plt.show()
    plt.close()

def heatmap(df: pd.DataFrame): 
    plt.figure(figsize=(15, 6))
    sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title("Heatmap of Features")
    plt.show()
    plt.close()

def initial_plot(df: pd.DataFrame):

    print("== Heatmap of Features ==\n")
    heatmap(df)
    """
    print("== Pair Plot of Features ==\n")
    pair_plot(df)
    """
    print("== Box Plot of Features ==\n")
    box_plot(df)

    return 

def main(): 
    #scaler = StandardScaler()
    #filepath=config.ROTATION_SEGMENTATION_DIR

    #sg.segment_all_users(activity=activity, threshold=threshold, scaler=scaler, filepath=filepath)
    activity = "sphereActivity"
    threshold = 0.75
    scaler_name = "standard"
    #result_directory = "dbscan_results"
    result_directory = "dbscan_results_rotation"

    print(f"===== Processing activity: {activity}, threshold: {threshold}, scaler: {scaler_name} =====\n")

    filepath = config.ROTATION_SEGMENTATION_DIR + f"/right_singular_vector/{activity}/{scaler_name}/rsv_{activity}_{threshold}_{scaler_name}.csv"
    df_origin = pd.read_csv(filepath, index_col=None)
    
    df = df_origin.drop(columns=['Username', 'Adjective', 'LogNumber'])
    df = df.dropna()
    
    n_cols = len(df.columns)
    length = len(df)
    
    print(f"Number of columns: {n_cols}\nLength of DataFrame: {length}\n")
    print("=================================================\n")


    initial_plot(df)


    print("== DBSCAN Clustering Results ==\n")
    grid(df)

    eps = 0.25
    min_samples = 8
    best_labels = dbscan_clustering(df, eps=eps, min_samples=min_samples)
    df['Label'] = best_labels
    
    # Insert Label column after LogNumber column in df_origin
    lognumber_index = df_origin.columns.get_loc('LogNumber')
    df_origin.insert(lognumber_index + 1, 'Label', best_labels)

    save_dir = config.PROCESSED_DATA_DIR + f"/{result_directory}/{activity}/{scaler_name}/all_points/dbscan_{activity}_{threshold}_{scaler_name}_eps{eps}_minsample{min_samples}.csv"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    df_origin.to_csv(save_dir, index=False)

    df_no_outliers = df.drop(df[df['Label'] == -1].index)
    unique_labels = df_no_outliers['Label'].unique()


    df_substantives = get_substantives_dataframe(df=df_no_outliers, unique_labels=unique_labels)
    print("\n== Substantives DataFrame ==\n")
    print(df_substantives.head())

    save_dir = config.PROCESSED_DATA_DIR + f"/{result_directory}/{activity}/{scaler_name}/substantives/substantives_{activity}_{threshold}_{scaler_name}_eps{eps}_minsample{min_samples}.csv"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    df_substantives.to_csv(config.PROCESSED_DATA_DIR + f"/{result_directory}/{activity}/{scaler_name}/substantives/substantives_{activity}_{threshold}_{scaler_name}_eps{eps}_minsample{min_samples}.csv", index=False)

    df_adjectives = get_adjectives_dataframe(df=df_origin, unique_labels=unique_labels)
    print("\n== Adjectives DataFrame ==\n")
    print(df_adjectives.head())
    print("...\n")

    save_dir = config.PROCESSED_DATA_DIR + f"/{result_directory}/{activity}/{scaler_name}/adjectives/"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    df_adjectives.to_csv(save_dir + f"adjectives_{activity}_{threshold}_{scaler_name}_eps{eps}_minsample{min_samples}.csv.csv", index=False)

    return 

if __name__ == "__main__":
    print("=== DBSCAN clustering script started... ===\n")
    main()
    exit(1) 