from pathlib import Path 

BASE_PATH = "/Users/grims/Documents/Research/Tesi/ML_tesi"

DATA_DIR = BASE_PATH + "/data_logs"
RAW_DATA_DIR = DATA_DIR + "/raw"
PROCESSED_DATA_DIR = DATA_DIR + "/processed" 
TEST_DATA_DIR = DATA_DIR + "/test_dataset/datasets"

RESULTS_DIR = DATA_DIR + "/results"
SEGMENTATION_DIR = RESULTS_DIR + "/segmentation"
BOTH_SEGMENTATION_DIR = RESULTS_DIR + "/both_segmentation"
ROTATION_SEGMENTATION_DIR = RESULTS_DIR + "/rotation_segmentation"
POSITION_SEGMENTATION_DIR = RESULTS_DIR + "/position_segmentation"
CLUSTERING_DIR = RESULTS_DIR + "/clustering"
SEGMENT_NUMBER_DIR = SEGMENTATION_DIR + "/segments_number"

GRAPHS_DIR = SEGMENTATION_DIR + "/graphs"
HISTOGRAMS_DIR = GRAPHS_DIR + "/histograms"
ROTATION_HISTOGRAMS_DIR = ROTATION_SEGMENTATION_DIR + "/rotation_graphs/rotation_histograms"
POSITION_HISTOGRAMS_DIR = POSITION_SEGMENTATION_DIR + "/position_graphs/position_histograms"
BOTH_HISTOGRAMS_DIR = BOTH_SEGMENTATION_DIR + "/both_graphs/both_histograms"

RSV_DIR = SEGMENTATION_DIR + "/right_singluar_vector"

MIN_NUMBER_SAMPLES = 100 

FEATURES = ['Index', 'Timestamp','HeadPosX','HeadPosY','HeadPosZ','HeadRotX','HeadRotY','HeadRotZ','RightPosX','RightPosY','RightPosZ','RightRotX','RightRotY','RightRotZ','LeftPosX','LeftPosY','LeftPosZ','LeftRotX','LeftRotY','LeftRotZ']
ROTATION_FEATURES = ['Index', 'Timestamp', 'HeadRotX', 'HeadRotY', 'HeadRotZ', 'RightRotX', 'RightRotY', 'RightRotZ', 'LeftRotX', 'LeftRotY', 'LeftRotZ']
POSITION_FEATURES = ['Index', 'Timestamp', 'HeadPosX', 'HeadPosY', 'HeadPosZ', 'RightPosX', 'RightPosY', 'RightPosZ', 'LeftPosX', 'LeftPosY', 'LeftPosZ']


def find_file_from_username(username: str):
    """
    Find the file path for a given username in the raw data directory.
    
    Args:
        username (str): The username to search for.
        
    Returns:
        str: The file path if found, otherwise None.
    """
    user_dir = Path(RAW_DATA_DIR) / username
    if user_dir.exists() and user_dir.is_dir():
        files = list(user_dir.glob("*.csv"))
        if files:
            return str(files[0])  # Return the first CSV file found
    return None

def find_file_from_username_and_basepath(username: str, base_path: str):
    """
    Find the file path for a given username in the specified base path.
    
    Args:
        username (str): The username to search for.
        base_path (str): The base path to search within.
        
    Returns:
        str: The file path if found, otherwise None.
    """
    user_dir = Path(base_path) / username
    if user_dir.exists() and user_dir.is_dir():
        files = list(user_dir.glob("*.csv"))
        if files:
            return str(files[0])  # Return the first CSV file found
    return None

def get_all_files():
    """
    Get all CSV files in the raw data directory, excluding test_data subdirectory.
    
    Returns:
        list: A list of file paths for all CSV files found.
    """
    all_files = list(Path(RAW_DATA_DIR).glob("**/*.csv"))
    filtered_files = [file for file in all_files if "test_data" not in str(file)]
    return filtered_files

