import os 
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing.src import etl, config 

MIN_NUMBER_SAMPLES = config.MIN_NUMBER_SAMPLES 
PLT_WIDTH = 16
PLT_HEIGHT = 6

 
def getRandomColor():
        R = list(range(256))  #np.arange(256)
        B = list(range(256))
        G = list(range(256))
        R = np.array(R)/255.0
        G = np.array(G)/255.0
        B = np.array(B)/255.0
        #print(R)
        random.shuffle(R)   
        random.shuffle(G)
        random.shuffle(B)
        colors = []
        for i in range(256):
            colors.append((R[i], G[i], B[i]))        
        return colors 
 
def getRandomMarker():
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    #markers = ['s','o', '*']   
    random.shuffle(markers) 
    return markers  

def cluster_scatter_plot2D_labelled(df: pd.DataFrame):
        """
        Plot scatter plot with different colors and markers for each cluster.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data to plot
        """
        plt.figure(figsize=(10, 6))
        colors = getRandomColor()  
        markers = getRandomMarker() 
        cNum = df.iloc[:, 2].max()   
        for j in range(cNum+1):
            cluster_data = df[df.iloc[:, 2] == j]
            plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], color=colors[j%len(colors)], 
                       label=('C'+str(j)), marker=markers[j%len(markers)], s=20)  
            
        plt.xlabel("x values", fontsize=13)
        plt.ylabel("y values", fontsize=13)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(loc='lower left')
        

def plot_coordinates(df: pd.DataFrame):
    """
    Plot a DataFrame with two columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot
        x_col (str): Name of the column for x-axis (if None, uses first column)
        y_col (str): Name of the column for y-axis (if None, uses second column) 
        title (str): Title for the plot
    """
    if df.shape[1] < 2:
        print("DataFrame must have at least 2 columns")
        return
    
    # Use specified columns or default to first two columns
    x_column = df.columns[0] 
    y_column = df.columns[1] 
    
    plt.figure(figsize=(10,6))
    plt.scatter(df[x_column], df[y_column], marker='o', color='blue')
    
    plt.title("Plot of " + x_column + " vs " + y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()


def plot_segmentation_index(segmentationIndexList: list, threshold: float):
    """    
    Plot the segmentation index over time with a threshold line.
    Args:
        segmentationIndexList (list): List of segmentation indices to plot
        threshold (float): Threshold value for segmentation
    """

    plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
    plt.plot(range(len(segmentationIndexList)), segmentationIndexList, marker='.', markersize=2, linewidth=0.8, color='b', label='Segmentation Index')
    
    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    plt.title('Segmentation Index Over Time')
    plt.xlabel('Frames')
    plt.ylabel('Segmentation Index')
    plt.legend()
    
    # Customize grid with more vertical lines
    ax = plt.gca()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, len(segmentationIndexList), max(1, len(segmentationIndexList)//50)))
    ax.grid(True, which='minor', alpha=0.2, linestyle='--')
    ax.minorticks_on()
    
    plt.tight_layout()
    #plt.show()

def plot_features_by_log(df: pd.DataFrame, activity: str, log_number: int): 
    """
    Plot time for a given activity and log number.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot
        feature (list): List of features to plot
        activity (str): Activity to filter by
        log_number (int): Log number to filter by
    """
    
    plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
    
    for col in df.columns:
        if col != 'Timestamp':
            plt.plot(range(len(df)), df[col], label=col)

    plt.title(f'Feature Values Over Time - Activity: {activity}, Log Number: {log_number}')
    plt.xlabel('Frames')
    plt.ylabel('Feature Value')
    plt.legend()
    
    ax = plt.gca()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
def plot_histogram_on_feature(df: pd.DataFrame, feature: str, activity: str, threshold: float, scaler: str):
    """
    Plot histogram of a feature after scaling.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot
        feature (str): Feature to plot
        activity (str): Activity to filter by
        threshold (float): Threshold value for segmentation
        scaler (str): Scaler used for the feature
    """
    segments = df['NumberOfSegments']

    # Definisco i valori possibili 
    x_values = np.arange(segments.min(), segments.max() + 1)

    # Conto le frequenze usando 
    counts = segments.value_counts().sort_index()

    # Creo una Serie con tutti i valori possibili, rimepiendo i mancanti con 0
    counts_full = pd.Series(0, index=x_values)
    counts_full.update(counts)

    plt.figure(figsize=(PLT_WIDTH - 4, PLT_HEIGHT))
    plt.bar(counts_full.index, counts_full.values, width=0.8, color='skyblue', edgecolor='black')

    plt.xticks(x_values)
    plt.xlabel('Number of Segments')
    plt.ylabel('Number of Activities')
    plt.title(f'Histogram of Number of Segments - Activity: {activity}, Threshold: {threshold}, Scaler: {scaler}')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
def plot_histogram_of_segments(filepath: str, savepath: str):
    """
    Plot a histogram of the number of segments from the segmentation results CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing segmentation results.
        savepath (str): Directory path to save the histogram plot.
    """

    df = pd.read_csv(filepath)
    if df.empty:
        print(f"No data found in {filepath}.")
        return
    
    activity = df['Activity'].iloc[0]
    threshold = df['Threshold'].iloc[0]
    scaler = df['Scaler'].iloc[0]

    plot_histogram_on_feature(df=df, feature='NumberOfSegments', activity=activity, threshold=threshold, scaler=scaler)

    # Create directory structure if it doesn't exist
    os.makedirs(savepath, exist_ok=True)

    complete_path = savepath + "/segmentation_histogram_" + activity + "_" + str(threshold) + "_" + scaler + ".png"

    plt.savefig(complete_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_all_histograms(filepath: str, savepath: str):
    """
    Plot histograms for all segmentation results in the specified directory.
    
    Args:
        filepath (str): Directory path containing segmentation result CSV files.
    """
    
    for root, dirs, files in os.walk(filepath):
        for dir in dirs: 
            dir_path = os.path.join(root, dir)
            for file in os.listdir(dir_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(dir_path, file)
                    print(f"Plotting histogram for {file_path}")
                    plot_histogram_of_segments(filepath=file_path, savepath=savepath)
