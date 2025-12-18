"""
Shared ML pipeline components for the Telco Customer Churn project.
Updated for Classification as per Final Project requirements.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Classification Models
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =============================================================================
# Building blocks for preprocessing
# =============================================================================

# 1. Log Pipeline: For skewed financial data (MonthlyCharges, TotalCharges)
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
    StandardScaler(),
)

# 2. Categorical Pipeline: For text features (InternetService, Contract, etc.)
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

# 3. Default Numerical Pipeline: For standard numeric features (Tenure, SeniorCitizen)
default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

def build_preprocessing():
    """
    Return the ColumnTransformer for the Telco Customer Churn dataset.
    """
    preprocessing = ColumnTransformer(
        [
            # Financial columns benefit from log scaling
            ("log", log_pipeline, ["monthlycharges", "totalcharges"]),
            
            # Encode all text/object columns
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        # Everything else (tenure, seniorcitizen) gets standard scaling
        remainder=default_num_pipeline,
    )
    return preprocessing


# =============================================================================
# Estimator factory - UPDATED FOR CLASSIFICATION
# =============================================================================

def make_estimator_for_name(name: str):
    """
    Given a model name, return an unconfigured CLASSIFIER instance.
    """
    if name == "ridge":
        return RidgeClassifier(random_state=42)
    
    elif name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=42)
    
    elif name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic", # Required for binary classification
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
    
    elif name == "lightgbm":
        return LGBMClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=-1
        )
    else:
        raise ValueError(f"Unknown classification model name: {name}")