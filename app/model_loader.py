from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).parents[1] / "artifacts" / "trained_models" / "xgboost_regression.joblib"


def get_pipeline():
    """Load the trained pipeline."""
    return joblib.load(MODEL_PATH)
