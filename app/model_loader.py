from pathlib import Path
import joblib

MODEL_PATH = Path("artifacts/trained_models/xgboost_regression.joblib")

model = joblib.load(MODEL_PATH)
