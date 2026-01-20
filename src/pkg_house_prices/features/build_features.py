from pathlib import Path
import pandas as pd
from pkg_house_prices.utils.config import CONFIG
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.data.data_cleaner import train_processed, test_processed

def one_hot_encode_train(df, categorical_cols, drop_first=True):
    """
    One-hot encode training data and store the columns for later use on test data.
    """
    logger.info("one_hot_encode_train() - One-hot encoding training data...")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    logger.info(f"one_hot_encode_train() - Encoded training data shape: {df_encoded.shape}")
    return df_encoded

def one_hot_encode_test(df, categorical_cols, train_columns, drop_first=True):
    """
    One-hot encode test data using the same columns as training data.
    """
    logger.info("one_hot_encode_test() - One-hot encoding test data...")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    
    # Align test data columns with training data columns
    df_encoded = df_encoded.reindex(columns=train_columns, fill_value=0)
    
    logger.info(f"one_hot_encode_test() - Encoded test data shape: {df_encoded.shape}")
    return df_encoded

# throw a message if the target variable is missing in training data
target_variable = CONFIG["data"]["target"]
if target_variable not in train_processed.columns:
    logger.error(f"Target variable '{target_variable}' not found in training data columns.")
    raise KeyError(f"Target variable '{target_variable}' not found in training data columns.")

# throw a message if the target variable is a categorical variable
if train_processed[target_variable].dtype == 'object':
    logger.error(f"Target variable '{target_variable}' is categorical. It should be numerical.")
    raise TypeError(f"Target variable '{target_variable}' is categorical. It should be numerical.")

categorical_columns = train_processed.select_dtypes(include=['object']).columns.tolist()
train_final = one_hot_encode_train(train_processed, categorical_columns)
test_final = one_hot_encode_test(test_processed, categorical_columns, train_final.columns)



