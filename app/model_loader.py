from pathlib import Path

from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.helpers import load_ml_model

MODEL_PATH = Path(__file__).parents[1] / "artifacts" / "trained_models" / "xgboost_regression.joblib"

model_bundle = None  # Global variable to hold the loaded model


# ------    -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def get_model():

    logger.info("get_model() - Loading model ...")
    global model_bundle
    if model_bundle is None:
        # Load the bundle saved by save_model()
        model_bundle = load_ml_model(MODEL_PATH)

    # Return the model
    return model_bundle
