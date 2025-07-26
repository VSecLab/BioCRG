import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from data_processing.src import config

ALPHA = 0.4

def compute_custom_index_df(df, labels, centroids_df, alpha):
    inter_cluster_dist = 0.0
    intra_cluster_dist = 0.0
    K = len(centroids_df)

    # Inter-cluster distance (tra centroidi)
    for i in range(K):
        for j in range(K):
            if i != j:
                dist = np.linalg.norm(centroids_df.iloc[i] - centroids_df.iloc[j])
                inter_cluster_dist += dist

    # Intra-cluster distance (punti -> centroide)
    for i in range(K):
        cluster_points = df[labels == i]
        centroid = centroids_df.iloc[i]
        distances = cluster_points.apply(lambda row: np.linalg.norm(row - centroid), axis=1)
        intra_cluster_dist += distances.sum()

    I = alpha * inter_cluster_dist + (1 - alpha) * intra_cluster_dist
    return I

def kmeans_clustering(df: pd.DataFrame, K: int):
    """
    Perform KMeans clustering on the DataFrame and save the results.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to cluster.
        K (int): The number of clusters to form.
    Returns:
        np.ndarray: An array of cluster labels for each data point.
    """
    
    kmeans = KMeans(n_clusters=K, random_state=42).fit(df)
    
    return kmeans.labels_

def grid(df: pd.DataFrame):
    k_to_test = range(1, 20)
    

    inertia = []
    index_values = []

    print(" ITER| INFO%s | CLUS |   Cluster Size" % (" "*14))
    print("-"*50)

    for i in k_to_test:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df)

        number_of_clusters = len(set(kmeans.labels_[kmeans.labels_ >= 0]))
        unique_labels = set(kmeans.labels_)
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:  # Exclude noise points
                cluster_sizes[label] = len(kmeans.labels_[kmeans.labels_ == label])

        if cluster_sizes and number_of_clusters > 1:
            cluster_info = ", ".join([f"C{label}: {size}" for label, size in sorted(cluster_sizes.items())])
            #print("    | Cluster sizes: %s" % cluster_info)
        else: 
            cluster_info = "None"
        
        print(" %3d | Tested with K = %3s| %3s  | %5s" % (i, i, number_of_clusters, cluster_info))

        labels = pd.Series(kmeans.labels_, index=df.index)
        centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)

        I = compute_custom_index_df(df, labels, centroids_df, ALPHA)
        index_values.append((i, I))

        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.xlabel('K')
    plt.ylabel('Sum of squared error')
    plt.plot(k_to_test, inertia, linewidth=2, marker='8')
    plt.show()
    plt.close()

    return index_values


def main(): 

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

    index_values = grid(df=df)
    # Visualizzazione
    ks, Is = zip(*index_values)
    plt.plot(ks, Is, marker='o')
    plt.title(f"Indice I vs K (alpha = {ALPHA})")
    plt.xlabel("Numero di cluster (K)")
    plt.ylabel("Indice I")
    plt.grid(True)
    plt.show()

    print("\n== Risultati dell'indice I per K ==\n")
    best_k, best_I = min(index_values, key=lambda x: x[1])
    print(f"K ottimale: {best_k} con I = {best_I:.4f}")

    K = best_k
    print(f"\n== Esecuzione KMeans con K={K} ==\n")

    best_labels = kmeans_clustering(df=df, K=K)

    print(f"Best labels for K={K}:\n{best_labels}\n")

    return 

if __name__ == "__main__":
    print("=== KMEANS clustering script started... ===\n")
    main()
    exit(1)