"""
risk_index.py

Computes the final combined risk index for NEO analysis by integrating:
- supervised hazard probability
- unsupervised anomaly score
- engineered physical risk score

Also provides utilities for normalization and ranking high‑risk objects.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Normalization
# -----------------------------

def normalize_components(df):
    """
    Normalize hazard probability, anomaly value, and engineered risk score
    into 0–1 ranges using MinMax scaling.

    Returns a new dataframe with:
        norm_hazard_prob
        norm_anomaly
        norm_risk_score
    """
    df = df.copy()
    scaler = MinMaxScaler()

    df["norm_hazard_prob"] = scaler.fit_transform(df[["hazard_probability"]])
    df["norm_anomaly"] = scaler.fit_transform(df[["anomaly_value"]])
    df["norm_risk_score"] = scaler.fit_transform(df[["risk_score"]])

    return df


# -----------------------------
# Final Risk Index
# -----------------------------

def compute_final_risk_index(df):
    """
    Compute the final combined risk index using weighted components:

        0.5 * supervised hazard probability
        0.3 * inverted anomaly score (lower anomaly_value = riskier)
        0.2 * engineered physical risk score

    Returns a dataframe with final_risk_index added.
    """
    df = df.copy()

    if not all(col in df.columns for col in [
        "norm_hazard_prob", "norm_anomaly", "norm_risk_score"
    ]):
        raise ValueError("Normalized columns missing. Run normalize_components() first.")

    df["final_risk_index"] = (
        0.5 * df["norm_hazard_prob"] +
        0.3 * (1 - df["norm_anomaly"]) +
        0.2 * df["norm_risk_score"]
    )

    return df


# -----------------------------
# Ranking
# -----------------------------

def get_top_risk_objects(df, n=20):
    """
    Return the top N highest‑risk objects based on final_risk_index.
    """
    if "final_risk_index" not in df.columns:
        raise ValueError("final_risk_index missing. Run compute_final_risk_index() first.")

    return df.sort_values("final_risk_index", ascending=False).head(n)
