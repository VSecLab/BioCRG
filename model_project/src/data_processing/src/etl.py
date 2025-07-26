import pandas as pd 
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, RobustScaler, StandardScaler

def load_data(file_path: str): 
    """
    Load data from a CSV file into a pandas DataFrame.
    Automatically handles non-numeric values by converting them to NaN and ensuring correct data types.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data as a DataFrame with correct numeric types.
    """

    try:
        df = pd.read_csv(file_path, index_col=None)
        
        # le colonne numeriche sono tutte quelle che non sono 'Activity'
        numeric_columns = [col for col in df.columns if col not in ['Activity', 'Username']]
        
        # converto le colonne numeriche in tipo float, gestendo gli errori
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Data successfully loaded - Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  

def filter_data_on_features(df: pd.DataFrame, features: list): 
    """
    Filter the DataFrame to include only specified features.
    
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        features (list): List of features to keep in the DataFrame.
        
    Returns:
        pd.DataFrame: The filtered DataFrame with only the specified features.
    """
    
    try:
        filtered_df = df[features]
        return filtered_df
    except KeyError as e:
        print(f"Error filtering data: {e}")
        return pd.DataFrame() 
    

def filter_data_on_activity(df: pd.DataFrame, activity: str):
    """
    Filter the DataFrame to include only rows with a specific activity.
    
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        activity (str): The activity to filter by.
        
    Returns:
        pd.DataFrame: The filtered DataFrame containing only the specified activity.
    """
    
    try:
        filtered_df = df[df['Activity'] == activity]
        return filtered_df
    except RuntimeError as e:
        print(f"Error filtering data on activity: {e}")
        return pd.DataFrame()
    
def filter_data_on_log_number(df: pd.DataFrame, log_number: int):
    """
    Filter the DataFrame to include only rows with a specific log number.
    
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        log_number (str): The log number to filter by.
        
    Returns:
        pd.DataFrame: The filtered DataFrame containing only the specified log number.
    """
    
    try:
        filtered_df = df[df['LogNumber'] == log_number]
        return filtered_df
    except KeyError as e:
        print(f"Error filtering data on log number: {e}")
        return pd.DataFrame()
    
def save_data(df: pd.DataFrame, file_path: str):
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path where the DataFrame should be saved.
    """
    
    try:
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def scaler_on_postion_and_rotation(df: pd.DataFrame, position_features: list, rotation_features: list, scaler_type='standard'): 
    """
    Normalize position and rotation features in the DataFrame using specified scaler type.
    Args:
        df (pd.DataFrame): The DataFrame containing the features to normalize.
        position_features (list): List of position feature names to normalize.
        rotation_features (list): List of rotation feature names to normalize.
        scaler_type (str): Type of scaler to use ('robust', 'minmax', 'quantile', 'standard'). Default is 'standard'.
    """

    if(scaler_type == 'robust'): 
        rotation_scaler = RobustScaler()
        position_scaler = RobustScaler()
    elif(scaler_type == 'minmax'):
        rotation_scaler = MinMaxScaler(feature_range=(-1, 1))
        position_scaler = MinMaxScaler(feature_range=(-1, 1))
    elif(scaler_type == 'quantile'):
        rotation_scaler = QuantileTransformer(output_distribution='uniform')
        position_scaler = QuantileTransformer(output_distribution='uniform')
    elif(scaler_type == 'standard'):
        rotation_scaler = StandardScaler()
        position_scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}. Supported types are 'robust', 'minmax', 'quantile', 'standard'.")
    
    rotation_cols = [col for col in rotation_features if col not in ['Timestamp', 'Index']]
    position_cols = [col for col in position_features if col not in ['Timestamp', 'Index']]
    
    #print(f"Rotation features to normalize: {rotation_cols}")
    #print(f"Position features to normalize: {position_cols}")
    
    # Normalize rotation features
    if rotation_cols:
        df[rotation_cols] = rotation_scaler.fit_transform(df[rotation_cols])
        print("Rotation features normalized")
    
    # Normalize position features  
    if position_cols:
        df[position_cols] = position_scaler.fit_transform(df[position_cols])
        print("Position features normalized")

    return df