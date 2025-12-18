from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="Telco Customer Churn API",
    description="API for predicting customer churn using an ML pipeline.",
    version="1.0"
)


MODEL_PATH = "models/global_best_model_optuna.pkl"

# 2. Debugging: Check if file exists inside Docker
if os.path.exists(MODEL_PATH):
    print(f"✓ Found model at: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print(f"❌ File not found at: {MODEL_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of 'models' folder: {os.listdir('models') if os.path.exists('models') else 'models folder missing'}")
    model = None

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define the input data structure using Pydantic
class CustomerData(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float

@app.get("/")
def home():
    return {"message": "Telco Churn API is running! Go to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input data to DataFrame
    input_data = pd.DataFrame([customer.dict()])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # Probability of Churn (1)
        
        result = "Churn" if prediction == 1 else "No Churn"
        
        return {
            "prediction": result,
            "churn_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Run locally if executed as script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)