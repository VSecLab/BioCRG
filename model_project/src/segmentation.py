import os
import numpy as np 
import pandas as pd 
from numpy.linalg import svd 
import adjectives_handle as ah
import matplotlib.pyplot as plt
from data_processing.src import etl
from data_processing.src import config 
import data_processing.src.plot as plot
 
MIN_NUMBER_SAMPLES = config.MIN_NUMBER_SAMPLES 
PLT_WIDTH = 16
PLT_HEIGHT = 6

def plot_segmentation_for_all_logs(df: pd.DataFrame, activity: str, features: list, threshold: float):
    """
    Plot segmentation index for all log numbers overlapped in a single plot.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot
        activity (str): Activity to filter by
        features (list): List of features to include
        threshold (float): Threshold value for segmentation
    """
    unique_log_numbers = sorted(df['LogNumber'].unique())
    print(f"Found {len(unique_log_numbers)} unique log numbers: {unique_log_numbers}")
    
    # Setup the plot
    plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
    
    # Generate a color map for different log numbers
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_log_numbers)))
    
    for i, log_number in enumerate(unique_log_numbers):
        # Filter data for current log number
        df_log = etl.filter_data_on_log_number(df, log_number=log_number)
        df_log = etl.filter_data_on_features(df_log, features=features)
        
        if df_log.empty or df_log.shape[0] < MIN_NUMBER_SAMPLES:
            print(f"No data found for log number {log_number}")
            continue
        
        # Reset index to start from 0
        df_log = df_log.reset_index(drop=True)
        
        print(f"Processing log number {log_number}, shape: {df_log.shape}")
        
        # Compute segmentation index
        segmentation_index_list, numberOfSegments, _, _, _ = segmentation(df_log, threshold)
        
        # Plot with different color for each log number
        plt.plot(range(len(segmentation_index_list)), segmentation_index_list, 
                marker='.', 
                markersize=1.5, 
                linewidth=0.8, 
                color=colors[i], 
                label=f'Log {log_number}',
                alpha=0.8)
    
    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    # Customize the plot
    plt.title(f'Segmentation Index Over Time - All Log Numbers\nActivity: {activity}')
    plt.xlabel('Frames')
    plt.ylabel('Segmentation Index')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Customize grid
    ax = plt.gca()
    ax.grid(True, alpha=0.3)
    ax.grid(True, which='minor', alpha=0.2, linestyle='--')
    ax.minorticks_on()
    
    plt.tight_layout()

def plot_seg(username: str, features: list, activity: str, threshold: float, scaler: str): 

    print("Starting segmentation test...\n")

    file_path = config.find_file_from_username(username=username)
    features = ['Timestamp','HeadPosX','HeadPosY','HeadPosZ','HeadRotX','HeadRotY','HeadRotZ','RightPosX','RightPosY','RightPosZ','RightRotX','RightRotY','RightRotZ','LeftPosX','LeftPosY','LeftPosZ','LeftRotX','LeftRotY','LeftRotZ']
    
    base_df = etl.load_data(file_path)

    # Normalizzo position e rotation separatamente e solo sull'attività specificata
    df = etl.filter_data_on_activity(df=base_df, activity=activity).dropna()
    df = etl.scaler_on_postion_and_rotation(df=df, position_features=config.POSITION_FEATURES, rotation_features=config.ROTATION_FEATURES, scaler_type=scaler)

    print(f"base_df after separate normalization: \n {df.tail()}\n")

    # ------------ #
    print("\nPlot logs\n")
    
    if df.empty:
        print(f"No data found for activity '{activity}'.")
        return

    df = df.reset_index(drop=True)    
    
    plot_segmentation_for_all_logs(df, activity, features[1:], threshold)

    # ------------ #
    """
    print("\nPlot value over time\n")
    log_number = 2

    df = etl.filter_data_on_log_number(df=df, log_number=log_number)
    df = etl.filter_data_on_features(df=df, features=features)

    if df.empty:
        print(f"No data found for activity '{activity}' and log number {log_number}.")
        return
    
    df = df.reset_index(drop=True)  

    #plot.plot_features_by_log(df, "sphereActivity", log_number)
    """
    plt.show()

def lsv_segmentation_plot(matrix: np.ndarray): 
    length = matrix.shape[0]
    
    # first left singular vector
    lfv = matrix[:, 0].reshape(-1, 1) 
    
    # plot the first left singular vector
    plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
    plt.plot(range(length), lfv, marker='.', markersize=1.5, linewidth=0.8, color='blue', alpha=0.7)
    
    # Add horizontal lines at 0.2 and -0.2
    plt.axhline(y=0.1, color='green', linestyle='-', linewidth=1, alpha=0.7, label='0.2')
    plt.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.7, label='0')
    plt.axhline(y=-0.1, color='purple', linestyle='-', linewidth=1, alpha=0.7, label='-0.2')
    
    plt.title("First Left Singular Vector Over Time")
    plt.xlabel("Lenght of the Left Singular Vector")
    plt.ylabel("First Left Singular Vector")
    ax = plt.gca()
    ax.grid(True, alpha=0.3)
    ax.grid(True, which='minor', alpha=0.2, linestyle='--')
    ax.minorticks_on()
    plt.tight_layout()
    plt.show()
    plt.close()
    return 

def segmentation(df: pd.DataFrame, threshold: float): 

    segmentationMatrix = np.empty((0, df.shape[1] - 1), float) 

    segmentationIndexList = []
    segmentationIndex = 0 

    right_singular_vector = np.empty((0, df.shape[1] - 1), float)
    lsv_dict = {}  # Dizionario per memorizzare i vettori singolari sinistri

    searchNewSegment = True  

    sigma1 = 0
    sigma2 = 0
    i = 0

    numberOfSegments = 1 

    for row in df.values: 
        segmentationMatrix = np.vstack([segmentationMatrix, row[1:]])
        
        U, S, Vh = svd(segmentationMatrix, full_matrices=True)
        
        sigma1 = S[0]
        if len(S) > 1:
            sigma2 = S[1]
        else: 
            sigma2 = 0

        segmentationIndex = sigma2 / sigma1

        if ((segmentationIndex > threshold) and searchNewSegment): 
            # quando supero la threshold e non sono in un nuovo segmento vuol dire che ho trovato un nuovo segmento
            right_singular_vector = np.vstack([right_singular_vector, Vh[0, :].reshape(1, -1)])

            # lsv_segmentation_plot(U)
            
            # Memorizza il vettore singolare sinistro nel dizionario
            lsv_dict["segNum_" + str(numberOfSegments)] = U[:, 0].copy()  # Salva il primo vettore singolare sinistro

            segmentationMatrix = np.empty((0, df.shape[1] - 1), float)
            segmentationMatrix = np.vstack([segmentationMatrix, row[1:]])
            searchNewSegment = False
            numberOfSegments += 1

        elif ((not searchNewSegment) and (segmentationIndex <= threshold)):
            # posso cercare un nuovo segmento se sono in un nuovo segmento e se scendo al di sotto della soglia in attesa di risalirci 
            searchNewSegment = True

        segmentationIndexList.append(segmentationIndex)
        i += 1

    # Aggiungo l'ultimo segmento se non è stato aggiunto
    right_singular_vector = np.vstack([right_singular_vector, Vh[0, :].reshape(1, -1)])
    # Memorizza il vettore singolare sinistro dell'ultimo segmento
    lsv_dict["segNum_" + str(numberOfSegments)] = U[:, 0].copy()

    return segmentationIndexList, numberOfSegments, right_singular_vector, lsv_dict

def segmentation_on_activity(file_path: str, features: list, activity: str, threshold: float, scaler: str):
    """
    Perform segmentation on the specified activity from the user's data file.
    
    Args:
        file_path (str): Path to the user's data file.
        activity (str): Activity to filter by.
        threshold (float): Threshold value for segmentation.
        scaler (str): Scaler type to use ('standard', 'minmax', etc.).
    Returns:
        tuple: A tuple containing:
            - logNumber_numberSegments: List of tuples with log number and number of segments.
            - rsv_on_activity: Right singular vectors for the activity.
            - lsv_on_activity: Dictionary with log numbers as keys and LSV dictionaries as values.
    """

    rsv_on_activity = np.empty((0, len(features)), float) 
    logNumber_numberSegments = []
    lsv_on_activity = {}  # Dizionario per memorizzare i LSV per ogni log

    base_df = etl.load_data(file_path)
    # Normalizzo position e rotation separatamente e solo sull'attività specificata
    df = etl.filter_data_on_activity(df=base_df, activity=activity).dropna()
    df = etl.scaler_on_postion_and_rotation(df=df, position_features=config.POSITION_FEATURES, rotation_features=config.ROTATION_FEATURES, scaler_type=scaler)
    
    if df.empty:
        print(f"No data found for activity '{activity}'.")
        return

    df = df.reset_index(drop=True)  

    unique_log_numbers = sorted(df['LogNumber'].unique())

    for i, log_number in enumerate(unique_log_numbers):
        
        # Filter data for the current log number and features
        df_log = etl.filter_data_on_log_number(df, log_number=log_number)
        df_log = etl.filter_data_on_features(df_log, features=features)
        
        if df_log.empty or df_log.shape[0] < MIN_NUMBER_SAMPLES:
            print(f"No data found for log number {log_number}")
            continue
        
        # Reset index to start from 0
        df_log = df_log.reset_index(drop=True)
        
        _, numberOfSegments, rsv, lsv_dict = segmentation(df_log, threshold)

        #print(f"Number of segments detected for log number {log_number}: {numberOfSegments}\n")
        logNumber_numberSegments.append((log_number.item(), numberOfSegments))
        
        # Aggiungi il dizionario LSV per questo log number
        lsv_on_activity["logNum_" + str(log_number.item())] = lsv_dict

        # Create array with log_number as first column and RSV data as remaining columns
        log_number_column = np.full((rsv.shape[0], 1), log_number.item())
        rsv_with_log = np.hstack([log_number_column, rsv[:, :]])
        rsv_on_activity = np.vstack([rsv_on_activity, rsv_with_log])
    
    return logNumber_numberSegments, rsv_on_activity, lsv_on_activity
    
def segment_all_users(activity: str, features: list, threshold: float, scaler: str, filepath: str): 
    """ 
    Segmentation for all users in the raw data directory and save results to a CSV file.
    Args:
        activity (str): Activity to filter by.
        threshold (float): Threshold value for segmentation.
        scaler (str): Scaler type to use ('standard', 'minmax', etc.).
        filepath (str): Directory path to save the segmentation results.
    Returns:
        tuple: A tuple containing:
            - rsv_df: DataFrame containing the right singular vectors and adjectives for all users.
            - lsv_all_users: Dictionary with usernames as keys and LSV activity dictionaries as values.
    """
    rsv_all_users = np.empty((0, len(features) - 1), float)
    rsv_usernames = []
    segmentation_list = []
    lsv_all_users = {}  # Dizionario per memorizzare i LSV per tutti gli utenti
    
    all_files = config.get_all_files()
    
    if not all_files:
        print("No files found in the raw data directory.")
        return
    
    for file_path in all_files:
        username = file_path.parent.name

        print(f"===== Processing user: {username}, act: {activity}, scaler: {scaler}, threshold: {threshold} =====")
        try: 
            logNumber_numberSegments, rsv, lsv_on_activity = segmentation_on_activity(file_path=file_path, features=features[1:], activity=activity, threshold=threshold, scaler=scaler)
            for log_number, number_of_segments in logNumber_numberSegments:
                print(f"Log Number: {log_number}, Number of Segments: {number_of_segments}")
                segmentation_list.append((username, activity, threshold, scaler, log_number, number_of_segments))

            # Aggiungi il dizionario LSV per questo utente
            lsv_all_users[username] = lsv_on_activity
            
            rsv_all_users = np.vstack([rsv_all_users, rsv[:, :]])
            # Add username for each row in rsv
            rsv_usernames.extend([username] * rsv.shape[0])
        except Exception as e:
            print(f"User {username} failed with error: {e}")
        

    df = pd.DataFrame(segmentation_list, columns=['Username', 'Activity', 'Threshold', 'Scaler', 'LogNumber', 'NumberOfSegments'])

    complete_filepath = filepath + f"/segments_number/{activity}/{scaler}" + f"/segmentation_results_{activity}_{threshold}_{scaler}.csv"
    rsv_path = filepath + f"/right_singular_vector/{activity}/{scaler}" + f"/rsv_{activity}_{threshold}_{scaler}.csv"

    directory = os.path.dirname(complete_filepath)
    os.makedirs(directory, exist_ok=True)
    rsv_directory = os.path.dirname(rsv_path)
    os.makedirs(rsv_directory, exist_ok=True)

    rsv_df = pd.DataFrame(rsv_all_users, columns=['LogNumber'] + [str(i) for i in range(1, len(features) - 1)])
    rsv_df.insert(0, 'Username', rsv_usernames)

    # Create the final dataframe with adjectives
    final_data = []
    
    # Process each RSV and its corresponding LSV
    for idx, row in rsv_df.iterrows():
        username = row['Username']
        log_number = int(row['LogNumber'])
        rsv_values = row.iloc[2:].values  # Get the 18 RSV values
        
        # Find the corresponding LSV for this RSV
        log_key = f"logNum_{log_number}"
        if username in lsv_all_users and log_key in lsv_all_users[username]:
            # Count segments from the same user and log to determine segment number
            user_log_count = 0
            for prev_idx in range(idx):
                if (rsv_df.iloc[prev_idx]['Username'] == username and 
                    rsv_df.iloc[prev_idx]['LogNumber'] == log_number):
                    user_log_count += 1
            
            segment_number = user_log_count + 1
            seg_key = f"segNum_{segment_number}"
            
            if seg_key in lsv_all_users[username][log_key]:
                lsv_vector = lsv_all_users[username][log_key][seg_key]
                print(f"Processing LSV for {username}, {log_key}, {seg_key}")
                
                # Segment the LSV using ah.segment_lsv()
                segmented_lsv = ah.segment_lsv(lsv_vector)

                #print(f"Segmented LSV: {segmented_lsv}\n")
                
                # For each category, process each segment separately
                for category, segments in segmented_lsv.items():
                    if segments:  # Only process if the category has segments
                        for segment in segments:  # Process each segment individually
                            if len(segment) > 0:
                                segment_mean = np.mean(segment)
                                print(f"Username: {username}, logNum: {log_number}, segNum: {seg_key}, Category: {category}, Segment Mean: {segment_mean}\n")
                                #print(f"Segment: {segment}\n")
                                
                                # Create a row for this segment mean
                                row_data = [username, segment_mean, log_number] + rsv_values.tolist()
                                final_data.append(row_data)
    
    # Create the final dataframe
    columns = ['Username', 'Adjective', 'LogNumber'] + [f'RSV_{i}' for i in range(1, len(features) - 1)]
    final_df = pd.DataFrame(final_data, columns=columns)

    #df.to_csv(complete_filepath, index=False, mode='a', header=False)
    #rsv_df.to_csv(rsv_path, index=False, mode='a', header=False)
    df.to_csv(complete_filepath, index=False)
    final_df.to_csv(rsv_path, index=False)

    #print(final_df)

    return final_df, lsv_all_users

def segment_everything(activity: list, features:list, threshold: list, scaler: list, filepath: str):
    """
    Perform segmentation for all combinations of activities, thresholds, and scalers.
    
    Args:
        activity (list): List of activities to filter by.
        threshold (list): List of threshold values for segmentation.
        scaler (list): List of scaler types to use ('standard', 'minmax', etc.).
        filepath (str): Directory path to save the segmentation results.
    """
    
    for act in activity:
        for thresh in threshold:
            for scal in scaler:
                if((scal == "quantile" and thresh > 0.5) or (scal in ["minmax", "standard", "robust"] and thresh < 0.5)):
                    continue
                print("\n==================================================\n")
                rsv_df, lsv_all_users = segment_all_users(activity=act, features=features, threshold=thresh, scaler=scal, filepath=filepath) 

def main(): 
    #activities = ["sphereActivity", "ladderActivity", "trashActivity", "pilotActivity"]
    activities = ["sphereActivity"]
    #thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    thresholds = [0.6]
    #scalers = ['standard', 'robust', 'quantile']
    scalers = ['standard']

    features = config.FEATURES
    filepath = config.BOTH_SEGMENTATION_DIR
    
    """segment_everything(activity=activities, features=features, threshold=thresholds, scaler=scalers, filepath=filepath)
    for activity in activities:
        os.makedirs(config.ROTATION_SEGMENTATION_DIR + f"/segments_number/{activity}", exist_ok=True)
        os.makedirs(config.ROTATION_HISTOGRAMS_DIR + f"/{activity}", exist_ok=True)
        plot.plot_all_histograms(filepath=config.ROTATION_SEGMENTATION_DIR + f"/segments_number/{activity}", savepath=config.ROTATION_HISTOGRAMS_DIR + f"/{activity}")
    """
    """filepath = config.find_file_from_username_and_basepath(username="grims3", base_path=config.RAW_DATA_DIR + "/test_data")
    print(f"File path: {filepath}")
    segmentation_on_activity(
        file_path=filepath, 
        features=features, 
        activity="sphereActivity", 
        threshold=0.75, 
        scaler="standard")"""
    
    #plot_seg(username="grims", features=features, activity="sphereActivity", threshold=0.65, scaler="standard")

def test_lsv(): 
    threshold = 0.6
    scaler = "standard"

    features = config.FEATURES
    filepath = config.BOTH_SEGMENTATION_DIR

    tmp = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/raw/grims/grims_log_20250719_1148_1YAYXWAD50.csv"
    base_df = etl.load_data(tmp)
    df = etl.filter_data_on_activity(df=base_df, activity="sphereActivity").dropna()
    df = etl.scaler_on_postion_and_rotation(df=df, position_features=config.POSITION_FEATURES, rotation_features=config.ROTATION_FEATURES, scaler_type=scaler)
    
    df = df.reset_index(drop=True)  

    unique_log_numbers = sorted(df['LogNumber'].unique())

    for i, log_number in enumerate(unique_log_numbers):
        
        # Filter data for the current log number and features
        df_log = etl.filter_data_on_log_number(df, log_number=log_number)
        df_log = etl.filter_data_on_features(df_log, features=features[1:])
        
        if df_log.empty or df_log.shape[0] < MIN_NUMBER_SAMPLES:
            print(f"No data found for log number {log_number}")
            continue
        
        # Reset index to start from 0
        df_log = df_log.reset_index(drop=True)
        
        _, n, _, _ = segmentation(df_log, threshold=threshold)
        print(f"Number of segments: {n}\n")

def test(): 
    threshold = 0.7
    scaler = "standard"
    target_activity = "sphereActivity"

    segmentation_result = f"/tmp"
    os.makedirs(segmentation_result, exist_ok=True)

    features = config.FEATURES
    filepath = config.BOTH_SEGMENTATION_DIR

    rsv_df, lsv_all_user = segment_all_users(
            activity=target_activity, 
            features=features,
            threshold=threshold, 
            scaler=scaler, 
            filepath=filepath
        )
    print(lsv_all_user)
    
if __name__ == "__main__":
    print("====== Segmentation Module ======")
    #main()

    test()
    exit(1)