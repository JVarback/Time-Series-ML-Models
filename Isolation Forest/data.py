from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import pandas as pd
import numpy as np
import random
import os

def getCSVsFromDirectory(directory: str) -> List[str]:
    """
    Retrieves all CSV file paths from the specified directory.

    Args:
    - directory: The path to the directory containing the CSV files.

    Returns:
    - A list of paths to the CSV files.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]


def splitFiles(file_paths: List[str], split_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Randomizes and splits the file paths into two lists based on the specified ratio.

    Args:
    - file_paths: The list of file paths to split.
    - split_ratio: The ratio for the first split (default is 0.8 for 80%/20% split).

    Returns:
    - A tuple of two lists: the first list for the first split, and the second list for the remaining files.
    """
    random.shuffle(file_paths)  # Randomize the order
    split_index = int(len(file_paths) * split_ratio)  # Find the index at which to split
    return file_paths[:split_index], file_paths[split_index:]


def loadDataForIsolationForest(file_paths, half_data = False, verbose = False):
    """
    Concatenates and normalizes data series matrix

    Args:
    - file_paths: The list of file paths.
    - half_data: A boolean indicating whether to keep only the first half of each sequence.

    Returns:
    - A numpy.NdArray: Concatenated matrix as time series
    """
    df_list = []

    for path in file_paths:
        df = pd.read_csv(path, skiprows=3)

        # Extract relevant columns
        machine_id_col = 'Link'
        axis_col = '_field'
        value_col = '_value'
        time_col = '_time'

        # Filter relevant columns
        filtered_df = df[[machine_id_col, axis_col, value_col, time_col]]

        # Pivot data to have separate columns for each Axis and its corresponding values
        pivoted_df = filtered_df.pivot_table(
            index=[machine_id_col, time_col], columns=[axis_col], values=value_col)

        # Reset the index to make Machine ID and Time as regular columns
        pivoted_df.reset_index(inplace=True)

        # Convert time column to datetime type
        pivoted_df[time_col] = pd.to_datetime(pivoted_df[time_col])

        # Sort DataFrame by Machine ID and Time
        pivoted_df.sort_values(by=[machine_id_col, time_col], inplace=True)

        # Fill missing values using forward fill
        pivoted_df.fillna(method='ffill', inplace=True)
        pivoted_df = pivoted_df.dropna()  # Drop rows with NaN values

        # Replace NaN with mean values
        # np.nan_to_num(pivoted_df, nan=np.nanmean(X))

        if half_data:
            if verbose:
                print("Halve sequence ...")
            pivoted_df = pivoted_df.groupby(machine_id_col).head(len(pivoted_df) // 2)

        # Drop Machine ID & Datetime
        time_series_data = pivoted_df.drop(
            [machine_id_col, time_col], axis=1).values

        df_list.append(time_series_data)

    # Combine time-series
    concatenated_time_series = np.concatenate(df_list)

    # Normalize
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(concatenated_time_series)

    return normalized_data

def extractRigRunFromCSVFileName(file_path: str) -> Tuple[int, int]:
    """
    Extract rig and run numbers from the CSV file name.

    Args:
    - file_path: Path to the CSV file.

    Returns:
    - A tuple containing rig and run numbers.
    """
    file_name = os.path.basename(file_path)
    parts = file_name.split(' ')
    rig_index = parts.index('rig')
    run_index = parts.index('run')
    rig_number = int(parts[rig_index + 1])
    run_number = int(parts[run_index + 1])
    return rig_number, run_number