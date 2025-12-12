from pathlib import Path
import pandas as pd
from pkg_house_prices.utils.config import CONFIG  # your YAML loader
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.read_config import read_config 

# --- Load data function ---
def _load_data():
    """
    Load train and test datasets using absolute paths from config
    """
    logger.info("_load_data() - Executing load data function...")
   
    # Logging paths
    train_path = read_config("data", "train")
    test_path  = read_config("data", "test")

    logger.info(f"_load_data() - Loading train data from {train_path}")
    logger.info(f"_load_data() - Loading test data from {test_path}")

   # Optional sanity check
    if not train_path.exists():
        logger.error(f"_load_data() - Train file not found: {train_path}")
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        logger.error(f"_load_data() - Test file not found: {test_path}")
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    return train, test

train, test = _load_data()

