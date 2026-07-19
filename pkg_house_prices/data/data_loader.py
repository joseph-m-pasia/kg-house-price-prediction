from importlib.resources import path

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from pkg_house_prices.utils.logger  import logger


# Data Pipeline Functions

# 1. Load data function

def load_data(data_path):
    """
    Load datasets using absolute paths from config
    """
    logger.info(f"load_data() - Loading data from {data_path}...")   

    data_path = Path(data_path)

    # Optional sanity check
    if not data_path.exists():
        logger.error(f"load_data() - Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dt = pd.read_csv(data_path)

    return dt

# 2. Split data between features and target variable function

def extract_features_target(data: pd.DataFrame, target_variable: str):
    """
    Load training data and split between features and target variable.
    Input: data (DataFrame), target_variable (str)
    Output: X (features), y (target variable)
    """
    logger.info("extract_features_target() - Extracting features and target variable...")

    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    return X, y

# 3. Split data between train and test sets function
def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42, transform_y=False)->tuple:
    """
    Split data into training and testing sets.

    Input: X (features), y (target variable), test_size (float), random_state (int), transform_y (bool)
    Output: X_train, X_test, y_train, y_test
    """

    logger.info("split_data() - Splitting data into training and testing sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info(f"split_data() - Completed data split: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    if transform_y:
        logger.info("split_data() - Transforming target variable to log scale...")
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    return X_train, X_test, y_train, y_test

# 4. Save data function
def save_data(X: pd.DataFrame, y: pd.Series, save_as: str, save_to: str)->None:
    """
    Save DataFrame to CSV file.
    Input: X (features), y (target variable), save_as (filename), save_to (filepath)
    Output: None
    """
    logger.info(f"save_data() - Saving data to {save_to}...")
    logger.info(f"save_data() - Data shape: {X.shape}, Target shape: {y.shape}")
    
    output_dir = Path(save_to)
    if not output_dir.exists():
        logger.info(f"save_data() - Creating directory {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / save_as

    data = pd.concat([X, y], axis=1)
    data.to_csv(output_path, index=False)
