import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📡", layout="centered")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# This path must match where Docker mounts the data volume
SCHEMA_PATH = Path("/app/data/data_schema.json")

# API_URL is set in docker-compose.yml
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Load schema from JSON file
# -----------------------------------------------------------------------------
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    # Fallback logic for local testing outside Docker
    if not path.exists():
        local_path = Path("../data/data_schema.json")
        if local_path.exists():
            path = local_path
            
    if not path.exists():
        st.error(f"❌ Schema file not found at: {path}")
        return {}

    with open(path, "r") as f:
        return json.load(f)

schema = load_schema(SCHEMA_PATH)

numerical_features = schema.get("numerical", {})
categorical_features = schema.get("categorical", {})

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("📡 Telco Customer Churn Predictor")
st.markdown(
    f"""
    This app predicts whether a customer is likely to **Churn** (leave the service).
    
    Backend API: `{API_BASE_URL}`
    """
)

st.header("Customer Information")

user_input: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# 1. Numerical Inputs (Sliders)
# -----------------------------------------------------------------------------
st.subheader("Subscription Details")

col1, col2 = st.columns(2)

# We manually arrange specific columns for a better UI
if "tenure" in numerical_features:
    stats = numerical_features["tenure"]
    user_input["tenure"] = col1.slider(
        "Tenure (Months)", 
        min_value=int(stats["min"]), 
        max_value=int(stats["max"]), 
        value=int(stats["mean"])
    )

if "monthlycharges" in numerical_features:
    stats = numerical_features["monthlycharges"]
    user_input["monthlycharges"] = col2.slider(
        "Monthly Charges ($)", 
        min_value=float(stats["min"]), 
        max_value=float(stats["max"]), 
        value=float(stats["mean"])
    )

if "totalcharges" in numerical_features:
    stats = numerical_features["totalcharges"]
    # Total charges is correlated with tenure * monthly, but we let user input it or estimate it
    default_total = user_input.get("tenure", 1) * user_input.get("monthlycharges", 50)
    user_input["totalcharges"] = st.slider(
        "Total Charges ($)", 
        min_value=float(stats["min"]), 
        max_value=float(stats["max"]), 
        value=float(default_total)
    )

# -----------------------------------------------------------------------------
# 2. Categorical Inputs (Dropdowns)
# -----------------------------------------------------------------------------
st.subheader("Demographics & Services")

# Create a multi-column layout for categorical features
cat_cols = list(categorical_features.keys())
# We split categorical features into two columns for compactness
left_column, right_column = st.columns(2)

for i, feature_name in enumerate(cat_cols):
    options = categorical_features[feature_name]
    
    label = feature_name.replace("_", " ").title()
    
    # Place in left or right column
    if i % 2 == 0:
        container = left_column
    else:
        container = right_column
        
    user_input[feature_name] = container.selectbox(
        label,
        options=options,
        key=feature_name
    )

# SeniorCitizen is often numeric (0/1) in data but treated as category in schema
# Ensure we pass it as int if the API expects int
if "seniorcitizen" in user_input:
    user_input["seniorcitizen"] = int(user_input["seniorcitizen"])

st.markdown("---")

# -----------------------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------------------
if st.button("🔮 Predict Churn Risk", type="primary"):
    
    # Show input for debugging
    with st.expander("View Input Payload"):
        st.json(user_input)

    with st.spinner("Analyzing Customer Data..."):
        try:
            resp = requests.post(PREDICT_ENDPOINT, json=user_input, timeout=30)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection Failed: {e}")
        else:
            if resp.status_code != 200:
                st.error(f"❌ API Error ({resp.status_code}): {resp.text}")
            else:
                result = resp.json()
                prediction = result.get("prediction", "Unknown")
                probability = result.get("churn_probability", 0.0)

                # Display Results
                st.markdown("### Prediction Results")
                
                if prediction == "Churn":
                    st.error(f"⚠️ **High Risk of Churn**")
                    st.metric("Churn Probability", f"{probability:.1%}", delta="-High Risk")
                else:
                    st.success(f"✅ **Customer likely to Stay**")
                    st.metric("Churn Probability", f"{probability:.1%}", delta="Safe")
                    
st.markdown("---")
st.caption("EAS 503 Final Project | Telco Customer Churn")