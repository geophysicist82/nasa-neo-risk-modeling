"""
feature_engineering.py

Contains all feature transformations used in the NEO risk analysis pipeline.
This module takes a raw dataframe and returns an enriched feature set.
"""

import numpy as np
import pandas as pd


def add_log_features(df):
    """
    Add log-transformed velocity and miss distance.
    """
    df = df.copy()
    df["log_velocity"] = np.log1p(df["relative_velocity"])
    df["log_miss_distance"] = np.log1p(df["miss_distance"])
    return df


def add_diameter_features(df):
    """
    Add diameter midpoint and kinetic proxy.
    """
    df = df.copy()
    df["diameter_mid"] = (df["est_diameter_min"] + df["est_diameter_max"]) / 2
    df["kinetic_proxy"] = df["diameter_mid"] * df["relative_velocity"]
    return df


def add_zscores(df):
    """
    Add z-score normalized velocity and distance.
    """
    df = df.copy()
    df["velocity_z"] = (df["relative_velocity"] - df["relative_velocity"].mean()) / df["relative_velocity"].std()
    df["distance_z"] = (df["miss_distance"] - df["miss_distance"].mean()) / df["miss_distance"].std()
    return df


def add_preliminary_risk(df):
    """
    Add the preliminary engineered risk score.
    """
    df = df.copy()
    df["risk_score"] = (
        df["velocity_z"] * 0.4 +
        df["log_miss_distance"] * -0.5 +
        df["diameter_mid"] * 0.1
    )
    return df


def build_features(df):
    """
    Full feature engineering pipeline.
    Applies all transformations in the correct order.
    """
    df = df.copy()
    df = add_log_features(df)
    df = add_diameter_features(df)
    df = add_zscores(df)
    df = add_preliminary_risk(df)
    return df
