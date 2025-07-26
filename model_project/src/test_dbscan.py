import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from data_processing.src import etl, config, plot

def pair_plot(df: pd.DataFrame):
    plt.figure(figsize=(15, 6))
    sns.pairplot(df)
    

def heatmap(df: pd.DataFrame): 
    plt.figure(figsize=(15, 6))
    sns.heatmap(df.corr(), annot=True)
    plt.title("Heatmap of Features")
    plt.xlabel("Features")
    plt.ylabel("Features")
    
def box_plot(df: pd.DataFrame):
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df, orient="h")
    plt.title("Box Plot of Features")
    plt.xlabel("Values")
    plt.ylabel("Features")


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
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(df)

    return db.labels_



def main(): 
    file = "/VDD_Heartshaped/dataset.csv"
    #file = "/Compound/dataset.csv"
    #file = "/ED_Hexagon/dataset.csv"
    #file = "/MDDM_D31/dataset.csv"
    #file = "/MDDM_G2/dataset.csv"
    #file = "/Aggregation/dataset.csv"
    #file = "/G50/dataset.csv" 

    # Load the dataset
    filename = config.TEST_DATA_DIR + file

    df = pd.read_csv(filename, index_col=None).dropna()

    if df.empty:
        print("Error: The dataset is empty or could not be loaded.")
        return
    print(f"Data successfully loaded. Shape: {df.shape}")
    
    if df.shape[1] == 2:
        plot.plot_coordinates(df)
    elif df.shape[1] == 3: 
        plot.cluster_scatter_plot2D_labelled(df)
    else:
        print("Error: The dataset does not have the expected shape for plotting.")
        return
    
    plt.show()
    plt.close() 
    
    heatmap(df[df.columns[:2]]) 
    pair_plot(df[df.columns[:2]])
    box_plot(df[df.columns[:2]])

    plt.show()
    plt.close() 
        
    # Estraiamo le coordinate per il clustering
    coordinates = df.iloc[:, 0:2].copy()
    true_labels = df.iloc[:, 2].copy() if df.shape[1] > 2 else None

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(coordinates)
    scaled_df = pd.DataFrame(scaled_array, columns=['X', 'Y'])

    box_plot(scaled_df)
    

    labels = dbscan_clustering(scaled_df, eps=0.5, min_samples=5)
    coordinates['Label'] = labels

    plt.show()
    plt.close()

    # plot.cluster_scatter_plot2D_labelled(coordinates)
    sns.scatterplot(data = coordinates, x='X', y='Y', hue='Label', palette='viridis', legend='full')
    plt.title("DBSCAN Clustering Results")
    plt.show()
    plt.close()

    return 

if __name__ == "__main__":
    main()