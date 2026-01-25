import joblib
import os
from pkg_house_prices.utils.logger import logger


# Load the model
def read_joblib(path, filename):
    logger.info(f"read_joblib() - Loading model from {path}/{filename} ...")
    full_path = os.path.join(path, filename)
    model = joblib.load(full_path)
    return model

