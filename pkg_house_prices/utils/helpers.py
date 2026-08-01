import joblib
import os
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.project_root import PROJECT_ROOT
from pkg_house_prices.utils.config import CONFIG


# Load the model
def load_ml_model(model_path: str = None):
    """
    Load a trained ML model from disk.
    Asumes the model is saved as a .pkl file using joblib.
    Args:    model_path (str): The path to the saved model file.
    Returns: The loaded model object, or None if the file does not exist.
    """
    logger.info("Loading model and its metrics...")

    if os.path.exists(model_path):
        ml_model = joblib.load(model_path)
        return ml_model
    else:
        logger.warning(f"Model not found at {model_path}. Returning None.")
        return None


def read_config(*keys):
    """
    Fetch nested YAML keys and return the values
    Example: read_config("data", "train")
    """
    logger.info(f"read_config() - Reading config keys: {keys}")
    logger.info(f"read_config() - Project Root is {PROJECT_ROOT}")

    d = CONFIG
    for k in keys:
        d = d[k]
    return PROJECT_ROOT / d
