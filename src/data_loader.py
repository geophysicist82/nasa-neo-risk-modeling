"""
data_loader.py

Handles loading raw NEO data from the data/raw directory.
Provides a clean, reusable function for all notebooks and modules.
"""

import pandas as pd
import os


def load_neo_data(path="../data/raw/neo.csv"):
    """
    Load the raw NEO dataset.

    Parameters
    ----------
    path : str
        File path to the raw NEO CSV.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with raw NEO data.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find dataset at: {path}")

    df = pd.read_csv(path)

    # Optional: enforce expected columns
    expected_cols = {
        "id", "name", "relative_velocity", "miss_distance",
        "absolute_magnitude", "est_diameter_min", "est_diameter_max",
        "hazardous"
    }

    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df
