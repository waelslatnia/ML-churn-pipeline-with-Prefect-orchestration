from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse
from pathlib import Path
import pandas as pd

app = FastAPI(
    title="ML Churn Prediction API",
    description="API pour prédire le churn des clients bancaires",
    version="1.0.0"
)


MODEL_PATH = "churn_model.pkl"
ENCODER_PATH = "gender_encoder.pkl"
SCALER_PATH = "scaler.pkl"

# Chargement du modèle, encodeur et scaler
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Modèle, encodeur et scaler chargés avec succès")
except FileNotFoundError as e:
    print(f"⚠️  Fichier manquant : {e}")
    model = None
    encoder = None
    scaler = None


class PredictionInput(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float

@app.get("/", response_class=HTMLResponse)
def serve_form():
    html_path = Path(__file__).parent / "churn_predictor.html"
    return html_path.read_text(encoding="utf-8")



@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None or encoder is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle ou preprocesseurs non chargés")

    try:
        # Création du DataFrame brut
        df = pd.DataFrame([input_data.model_dump()])

        # 1. Encodage du Gender
        df["Gender"] = encoder.transform(df["Gender"])

        # 2. One-hot encoding pour Geography (comme dans prepare_data)
        df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

        # 3. S'assurer que toutes les colonnes attendues sont présentes
        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        # 4. Scaling
        X = scaler.transform(df)

        # 5. Prédiction
        prediction = model.predict(X)
        probability = model.predict_proba(X)[0][1]

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "message": "Le client va quitter la banque" if prediction[0] == 1 else "Le client va rester"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
