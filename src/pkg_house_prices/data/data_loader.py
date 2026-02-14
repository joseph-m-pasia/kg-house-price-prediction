from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from pkg_house_prices.utils.config import CONFIG  # your YAML loader
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.helpers import read_config 

# --- Load data function ---
def _load_data(data_path):
    """
    Load datasets using absolute paths from config
    """
    logger.info("_load_data() - Executing load data function...")   

    # Optional sanity check
    if not data_path.exists():
        logger.error(f"_load_data() - Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    train = pd.read_csv(data_path)

    return train


# --- Main execution ---
if __name__ == "__main__":
    logger.info("data_loader.py - Starting data loading and splitting process...")              
    train_path = read_config("data", "train")
    train = _load_data(train_path)

    target_variable = CONFIG["data"]["target"]  
    X = train.drop(columns=[target_variable])
    y = train[target_variable]

    # Split data between training and testing
    logger.info("_split_data() - Splitting data into features and target variable...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["data"]["test_size"], random_state=42)  

    logger.info(f"_split_data() - Completed data split: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
