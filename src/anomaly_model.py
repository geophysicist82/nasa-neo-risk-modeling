"""
anomaly_model.py

Implements anomaly detection for NEO risk analysis using Isolation Forest.
Provides functions to fit the model, score anomalies, and extract top outliers.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest


# -----------------------------
# Core anomaly detection
# -----------------------------

def fit_isolation_forest(df, feature_cols, contamination=0.01, random_state=42):
    """
    Fit an Isolation Forest model on selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with engineered features.
    feature_cols : list
        List of column names to use for anomaly detection.
    contamination : float
        Proportion of anomalies to flag (default 1%).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    model : IsolationForest
        Trained Isolation Forest model.
    """
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(df[feature_cols])
    return model


def add_anomaly_scores(df, model, feature_cols):
    """
    Add anomaly predictions and anomaly values to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    model : IsolationForest
        Trained model.
    feature_cols : list
        Columns used for scoring.

    Returns
    -------
    pd.DataFrame
        Dataframe with anomaly_score and anomaly_value columns added.
    """
    df = df.copy()
    df["anomaly_score"] = model.predict(df[feature_cols])
    df["anomaly_value"] = model.decision_function(df[feature_cols])
    return df


# -----------------------------
# Utility: rank anomalies
# -----------------------------

def get_top_anomalies(df, n=20):
    """
    Return the top N most anomalous objects (lowest anomaly_value).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with anomaly_value column.
    n : int
        Number of anomalies to return.

    Returns
    -------
    pd.DataFrame
        Top N anomalies sorted by anomaly_value ascending.
    """
    if "anomaly_value" not in df.columns:
        raise ValueError("Dataframe must contain 'anomaly_value' before ranking anomalies.")

    return df.sort_values("anomaly_value").head(n)
