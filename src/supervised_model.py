"""
supervised_model.py

Implements the supervised hazard classification model for NEO risk analysis.
Uses logistic regression to estimate hazard probability and provides evaluation utilities.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# -----------------------------
# Model Training
# -----------------------------

def train_hazard_model(df, feature_cols, test_size=0.2, random_state=42):
    """
    Train a logistic regression model to predict NEO hazard classification.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with engineered features.
    feature_cols : list
        Columns used as predictors.
    test_size : float
        Proportion of data used for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels for test set.
    y_pred : np.ndarray
        Predicted class labels.
    y_prob : np.ndarray
        Predicted hazard probabilities.
    """

    X = df[feature_cols]
    y = df["hazardous"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_pred, y_prob


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_hazard_model(y_test, y_pred, y_prob):
    """
    Print classification metrics and ROC AUC.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    dict
        Dictionary containing metrics.
    """

    report = classification_report(y_test, y_pred, output_dict=True)
    roc = roc_auc_score(y_test, y_prob)

    return {
        "classification_report": report,
        "roc_auc": roc
    }


# -----------------------------
# Add hazard probability to df
# -----------------------------

def add_hazard_probability(df, model, feature_cols):
    """
    Add predicted hazard probability to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    model : LogisticRegression
        Trained model.
    feature_cols : list
        Columns used for prediction.

    Returns
    -------
    pd.DataFrame
        Dataframe with hazard_probability column added.
    """

    df = df.copy()
    df["hazard_probability"] = model.predict_proba(df[feature_cols])[:, 1]
    return df


# -----------------------------
# Coefficient Table
# -----------------------------

def get_coefficient_table(model, feature_cols):
    """
    Return a table of coefficients and odds ratios.

    Parameters
    ----------
    model : LogisticRegression
        Trained model.
    feature_cols : list
        Names of features.

    Returns
    -------
    pd.DataFrame
        Table of coefficients and odds ratios.
    """

    coef = model.coef_[0]
    return pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef,
        "odds_ratio": np.exp(coef)
    }).sort_values("odds_ratio", ascending=False)
