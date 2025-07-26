import os 
import warnings
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from data_processing.src import etl, config, plot
import segmentation as sg

# Suppress specific deprecation warnings from sklearn
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'", category=FutureWarning) 

def get_substantives_dict(df: pd.DataFrame, unique_labels: np.ndarray):
    """
    Create a dictionary mapping cluster labels to their centers.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to cluster.
    Returns:
        dict: A dictionary where keys are cluster labels and values are the centers of the clusters.
    """
    
    centers = df.groupby('Label').mean().values
    dict_sustantives = dict.fromkeys(unique_labels, None)

    for label in unique_labels:
        dict_sustantives[label] = centers[label]

    return dict_sustantives

def get_adjectives_dict(df: pd.DataFrame, unique_labels: np.ndarray):
    """
    Create a dictionary mapping cluster labels to adjectives.
    
    Args:
        df (pd.DataFrame): The original DataFrame containing the 'Adjective' column.
        best_labels (np.ndarray): The array of cluster labels.
        
    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of adjectives.
    """
        
    dict_adjectives = dict.fromkeys(unique_labels, None)
        
    temp = []
    
    for i in unique_labels: 
        temp = list(df.loc[df['Label'] == i, 'Adjective'])
        dict_adjectives[i] = temp
    
    return dict_adjectives

def grid(df: pd.DataFrame): 

    min_cluster_size_to_test = [round(min_cluster_size,1) for min_cluster_size in np.arange(2, 15, 1)]
    min_samples_to_test = range(3, 21, 1)

    # Dataframe per la metrica sulla distanza media dei noise points dai K punti più vicini
    results_noise = pd.DataFrame( 
        data = np.zeros((len(min_cluster_size_to_test),len(min_samples_to_test))), # Empty dataframe
        columns = min_samples_to_test, 
        index = min_cluster_size_to_test
    )

    # Dataframe per la metrica sul numero di cluster
    results_clusters = pd.DataFrame( 
        data = np.zeros((len(min_cluster_size_to_test),len(min_samples_to_test))), # Empty dataframe
        columns = min_samples_to_test, 
        index = min_cluster_size_to_test
    )

    results_outliers = pd.DataFrame(
        data = np.zeros((len(min_cluster_size_to_test),len(min_samples_to_test))), # Empty dataframe
        columns = min_samples_to_test, 
        index = min_cluster_size_to_test
    )

    i = 0

    print("ITER| INFO%s | CLUS    OUT   |   Cluster Size" % (" "*39))
    print("-"*84)

    for min_cluster_size in min_cluster_size_to_test:
        for min_samples in results_noise:
            
            i += 1
            
            # Calcolo le metriche
            cluster_metric, outliers_metric = scanning(df, min_cluster_size, min_samples, i)
            
            # Inserisco i risultati nei relativi dataframe
            results_clusters.loc[min_cluster_size, min_samples] = cluster_metric
            results_outliers.loc[min_cluster_size, min_samples] = outliers_metric


    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(18,8))
    print("\n== Results of the grid search ==\n")
    sns.heatmap(results_clusters, annot = True, ax = ax1, cbar = False).set_title("METRIC: Number of clusters")
    sns.heatmap(results_outliers, annot = True, ax = ax2, cbar = False).set_title("METRIC: Number of outliers")

    ax1.set_xlabel("MIN_SAMPLE"); ax2.set_xlabel("MIN_SAMPLE")
    ax1.set_ylabel("MIN_CLUSTER_SIZE"); ax2.set_ylabel("MIN_CLUSTER_SIZE")

    plt.tight_layout(); plt.show(); plt.close()

    return

def scanning(df: pd.DataFrame, min_cluster_size: float, min_samples: int, iter: int):


    dbscan_model_ = HDBSCAN(min_cluster_size=min_cluster_size, 
                            min_samples=min_samples, 
                            cluster_selection_method='eom',
                            cluster_selection_epsilon = 0.0,
                            allow_single_cluster=False,
                            metric='euclidean', 
                            algorithm='best',
                            leaf_size=30)
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

    print(" %3d | Tested with mcs = %3s and min_samples = %3s | %3s %7s   | %5s" % (iter, min_cluster_size, min_samples, number_of_clusters, number_of_outliers, cluster_info))
        
    return(number_of_clusters, number_of_outliers)
    
def hdbscan_clustering(df: pd.DataFrame, min_cluster_size: float = 3, min_samples: int = 5):
    """
    Perform HDBSCAN clustering on the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to cluster.
        min_cluster_size (float): The minimum size of clusters.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
        np.ndarray: An array of cluster labels, where -1 indicates noise.
    """
    db = HDBSCAN(min_cluster_size=min_cluster_size, 
                min_samples=min_samples, 
                cluster_selection_method='eom',
                allow_single_cluster=False,
                metric='euclidean',
                algorithm='best',
                leaf_size=20)
    
    db.fit(df)
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

def heatmap(df: pd.DataFrame): 
    plt.figure(figsize=(15, 6))
    sns.heatmap(df.corr(), annot=True)
    plt.title("Heatmap of Features")
    plt.show()
    plt.close()

def main(): 
    activity = "sphereActivity"
    threshold = 0.6
    scaler_name = "standard"

    print(f"===== Processing activity: {activity}, threshold: {threshold}, scaler: {scaler_name} =====\n")

    #scaler = StandardScaler()
    #filepath=config.SEGMENTATION_DIR

    #sg.segment_all_users(activity=activity, threshold=threshold, scaler=scaler, filepath=filepath)

    filepath = config.SEGMENTATION_DIR + f"/right_singular_vector/{activity}/{scaler_name}/rsv_{activity}_{threshold}_{scaler_name}.csv"
    df_origin = pd.read_csv(filepath, index_col=None)
    
    df = df_origin.drop(columns=['Username', 'Adjective', 'LogNumber'])
    df = df.dropna()
    
    n_cols = len(df.columns)
    length = len(df)
    print(f"Number of columns: {n_cols}\nLength of DataFrame: {length}\n")
    print("=================================================\n")

    print("== Heatmap of Features ==\n")
    #heatmap(df)
    """
    print("== Pair Plot of Features ==\n")
    pair_plot(df)

    print("== Box Plot of Features ==\n")
    box_plot(df)"""

    print("== DBSCAN Clustering Results ==\n")
    grid(df)

    best_labels = hdbscan_clustering(df, min_cluster_size=3, min_samples=4)
    df['Label'] = best_labels
    df_origin['Label'] = best_labels

    # Create subset with first 8 columns plus the Label column
    #df_subset = df.iloc[:, lambda df: [1, 4, 7, 16]].copy()
    """df_subset = df.iloc[:, :8].copy()
    df_subset['Label'] = df['Label']

    sns.pairplot(df_subset, hue='Label', palette='Set1')
    plt.suptitle("DBSCAN Clustering Results", y=1.02)
    plt.show()
    plt.close()"""

    df_no_outliers = df.drop(df[df['Label'] == -1].index)
    unique_labels = df_no_outliers['Label'].unique()

    dict_substantives = get_substantives_dict(df=df_no_outliers, unique_labels=unique_labels)
    print("\n== Substantives Dictionary ==\n")
    for label, center in dict_substantives.items():
        print(f"Label {label}: {center}")
    
    dict_adjectives = get_adjectives_dict(df=df_origin, unique_labels=unique_labels)
    print("\n== Adjectives Dictionary ==\n")
    for label, adjectives in dict_adjectives.items():
        print(f"Label {label}: {adjectives}")
    
    return 

if __name__ == "__main__":
    print("=== HDBSCAN clustering script started... ===\n")
    main()
    exit(1) 