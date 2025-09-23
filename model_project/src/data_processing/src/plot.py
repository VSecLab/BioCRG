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


def plot_three_columns(df: pd.DataFrame, col1=None, col2=None, col3=None, plot_type='line', title=None):
    """
    Plotta tre colonne di un DataFrame usando diversi tipi di grafico.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati da plottare
        col1 (str): Nome della prima colonna (se None, usa la prima colonna)
        col2 (str): Nome della seconda colonna (se None, usa la seconda colonna) 
        col3 (str): Nome della terza colonna (se None, usa la terza colonna)
        plot_type (str): Tipo di grafico ('line', 'scatter', 'bar', '3d_scatter')
        title (str): Titolo del grafico
    """
    
    if df.shape[1] < 3:
        print("Il DataFrame deve avere almeno 3 colonne")
        return
    
    # Usa le colonne specificate o default alle prime tre colonne
    x_column = col1 if col1 else df.columns[0]
    y_column = col2 if col2 else df.columns[1] 
    z_column = col3 if col3 else df.columns[2]
    
    # Verifica che le colonne esistano
    missing_cols = [col for col in [x_column, y_column, z_column] if col not in df.columns]
    if missing_cols:
        print(f"Colonne mancanti nel DataFrame: {missing_cols}")
        return
    
    if plot_type == '3d_scatter':
        # Plot 3D scatter
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df[x_column], df[y_column], df[z_column], 
                           c=range(len(df)), cmap='viridis', alpha=0.7, s=50)
        
        ax.set_xlabel(x_column, fontsize=12)
        ax.set_ylabel(y_column, fontsize=12)
        ax.set_zlabel(z_column, fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'3D Scatter Plot: {x_column} vs {y_column} vs {z_column}', fontsize=14)
            
        # Aggiunge una colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        
    else:
        # Plot 2D con tre linee/scatter/bar
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'line':
            plt.plot(range(len(df)), df[x_column], label=x_column, marker='o', markersize=3, linewidth=2)
            plt.plot(range(len(df)), df[y_column], label=y_column, marker='s', markersize=3, linewidth=2)
            plt.plot(range(len(df)), df[z_column], label=z_column, marker='^', markersize=3, linewidth=2)
            
        elif plot_type == 'scatter':
            # Per scatter plot, uso x_column come asse x e plotto y e z
            plt.scatter(df[x_column], df[y_column], label=f'{y_column} vs {x_column}', alpha=0.7, s=30)
            plt.scatter(df[x_column], df[z_column], label=f'{z_column} vs {x_column}', alpha=0.7, s=30)
            plt.xlabel(x_column, fontsize=12)
            
        elif plot_type == 'bar':
            x_pos = np.arange(len(df))
            width = 0.25
            
            plt.bar(x_pos - width, df[x_column], width, label=x_column, alpha=0.8)
            plt.bar(x_pos, df[y_column], width, label=y_column, alpha=0.8)
            plt.bar(x_pos + width, df[z_column], width, label=z_column, alpha=0.8)
            plt.xticks(x_pos, range(len(df)))
            
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title(f'Plot delle colonne: {x_column}, {y_column}, {z_column}', fontsize=14)
        
        if plot_type != 'scatter':
            plt.xlabel('Index', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_three_columns_subplots(df: pd.DataFrame, col1=None, col2=None, col3=None, title=None):
    """
    Plotta tre colonne di un DataFrame in subplot separati.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati da plottare
        col1 (str): Nome della prima colonna (se None, usa la prima colonna)
        col2 (str): Nome della seconda colonna (se None, usa la seconda colonna)
        col3 (str): Nome della terza colonna (se None, usa la terza colonna)
        title (str): Titolo generale del grafico
    """
    
    if df.shape[1] < 3:
        print("Il DataFrame deve avere almeno 3 colonne")
        return
    
    # Usa le colonne specificate o default alle prime tre colonne
    x_column = col1 if col1 else df.columns[0]
    y_column = col2 if col2 else df.columns[1]
    z_column = col3 if col3 else df.columns[2]
    
    # Verifica che le colonne esistano
    missing_cols = [col for col in [x_column, y_column, z_column] if col not in df.columns]
    if missing_cols:
        print(f"Colonne mancanti nel DataFrame: {missing_cols}")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot prima colonna
    axes[0].plot(range(len(df)), df[x_column], color='blue', linewidth=2, marker='o', markersize=2)
    axes[0].set_title(f'{x_column}', fontsize=12)
    axes[0].set_ylabel(x_column, fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot seconda colonna  
    axes[1].plot(range(len(df)), df[y_column], color='red', linewidth=2, marker='s', markersize=2)
    axes[1].set_title(f'{y_column}', fontsize=12)
    axes[1].set_ylabel(y_column, fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Plot terza colonna
    axes[2].plot(range(len(df)), df[z_column], color='green', linewidth=2, marker='^', markersize=2)
    axes[2].set_title(f'{z_column}', fontsize=12)
    axes[2].set_xlabel('Index', fontsize=11)
    axes[2].set_ylabel(z_column, fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'Plot separati delle colonne: {x_column}, {y_column}, {z_column}', fontsize=14)
    
    plt.tight_layout()
    plt.show()
