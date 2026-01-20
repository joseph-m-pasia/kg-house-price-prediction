from pathlib import Path
import pandas as pd
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.read_config import read_config 
from pkg_house_prices.data.data_loader import train, test
from pkg_house_prices.utils.config import CONFIG

# --- Preprocess data function ---
# Remove variables from datasets with missing data greater than threshold
def _preprocess_train(df):
    logger.info("_preprocess_train() - Executing preprocess function...")
    missing_threshold = CONFIG['params']['missing_threshold']
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index
    df = df.drop(columns=cols_to_drop)
    logger.info("_preprocess_train() - Executing drop columns...")

    # throw a log message if target variable has been dropped
    target_var = CONFIG['data']['target']
    if target_var in cols_to_drop:
        logger.warning(f"_preprocess_train() - Target variable '{target_var}' was dropped due to high missing ratio.")
    logger.info("_preprocess_train() - Returning processed dataframe...")
    return df, cols_to_drop

def _preprocess_test(df, cols_to_drop):
    logger.info("_preprocess_test() - Executing preprocess function...")
    logger.info("_preprocess_test() - Executing drop columns...")
    df = df.drop(columns=cols_to_drop)
    logger.info("_preprocess_test() - Returning processed dataframe...")
    return df

train_processed, cols_to_drop = _preprocess_train(train)
test_processed = _preprocess_test(test, cols_to_drop )