"""
This module generates the final predictions for the house prices prediction task using the champion model identified in the previous evaluation step. It loads the trained champion model from disk, makes predictions on the test set, transforms the predictions back to the original scale if necessary, and saves the results in a CSV file for submission or further analysis.
Author: Joseph M.P.

"""

import pandas as pd 
import numpy as np
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.helpers import read_joblib
from pkg_house_prices.utils.project_root import PROJECT_ROOT
from pkg_house_prices.utils.config import CONFIG
from pkg_house_prices.data.data_loader import X_test, y_test



# This script generates the final results for the house prices prediction task. It loads the trained model, makes predictions on the test set, and saves the results in a CSV file.

# Read the best model from the previous evaluation step
logger.info("task_04_generate_final_results.py - Loading the champion model and generating final predictions...")
champion_model_name = "XGBoost Regression"  # Replace with the actual best model name
output_path_model = CONFIG['models']['output_path']
champion_model = read_joblib(PROJECT_ROOT / output_path_model, "champion_xgboost_regression.joblib")  

# Read the test data
logger.info("task_04_generate_final_results.py - Loading test data for predictions...")
pd_read_csv_path = CONFIG['data']['test']  # Assuming you have a test dataset path in your config
test_data = pd.read_csv(PROJECT_ROOT / pd_read_csv_path)

# Make predictions on the test set
logger.info(f"task_04_generate_final_results.py - Making predictions with the champion model: {champion_model_name} ...")
y_pred = champion_model.predict(test_data)

# Transform predictions back to original scale if you took log transformation
logger.info("task_04_generate_final_results.py - Transforming predictions back to original scale...")
y_pred_original_scale = np.expm1(y_pred)  # Inverse of log1p

# Save the predictions to a CSV file
logger.info("task_04_generate_final_results.py - Saving final predictions to CSV...")
output_path_data = CONFIG['data']['final_output_path']  
results_df = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": y_pred_original_scale
})
results_df.to_csv(PROJECT_ROOT / output_path_data / "final_predictions.csv", index=False )