import os
import traceback
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

# =======================
# LOGGER SETUP
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnosis_api")

# =======================
# CONFIGURATION
# =======================
MODEL_DIR = r"C:\Users\junio\Desktop\Projects\med-reminder\med-assistant-backend\diagnosis-api\models"
MODEL_FILE = "disease_diagnosis_pipeline.pkl"

# =======================
# GLOBAL VARIABLES
# =======================
model = None

# =======================
# FASTAPI APP
# =======================
app = FastAPI(title="Disease Diagnosis API", version="2.0")

# =======================
# REQUEST MODEL
# =======================
class DiagnosisRequest(BaseModel):
    Age: int
    Gender: str
    Symptom_1: str
    Symptom_2: str
    Symptom_3: str
    Heart_Rate_bpm: float
    Body_Temperature_C: float
    Blood_Pressure_mmHg: str
    Oxygen_Saturation: float = Field(..., alias="Oxygen_Saturation_%")
    Severity: str
    Treatment_Plan: str

    class Config:
        allow_population_by_field_name = True  # allows internal + alias usage

# =======================
# STARTUP EVENT
# =======================
@app.on_event("startup")
async def load_model():
    """Load the ML pipeline at startup"""
    global model
    logger.info("🚀 Loading disease diagnosis model...")

    try:
        model_path = os.path.join(MODEL_DIR, MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded successfully from {model_path}")

        routes = [r.path for r in app.routes]
        logger.info("\n📡 Available Routes:\n" + "\n".join(f"➡️ {r}" for r in routes))

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        traceback.print_exc()

# =======================
# DEBUG ENDPOINT
# =======================
@app.get("/debug")
async def debug_model():
    """Check if model is loaded"""
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "model_path": os.path.join(MODEL_DIR, MODEL_FILE)
    }

# =======================
# PREDICTION ENDPOINT
# =======================
@app.post("/predict-diagnosis")
async def predict_diagnosis(request: DiagnosisRequest):
    """Predict disease diagnosis from patient data"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /debug endpoint.")

    try:
        # Convert request to DataFrame with original training column names
        input_df = pd.DataFrame([request.dict(by_alias=True)])
        logger.info(f"📥 Input received: {input_df.to_dict(orient='records')[0]}")

        # Predict directly using pipeline (handles encoding internally)
        prediction = model.predict(input_df)[0]

        logger.info(f"✅ Prediction: {prediction}")
        return {"prediction": str(prediction), "message": "Diagnosis predicted successfully"}

    except Exception as e:
        logger.error(f"💥 Prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
