"""
This module evaluates the performance of trained regression models (Linear, Ridge, Lasso, ElasticNet, XGBoost) on the test set.
It loads the trained models from disk, makes predictions on the test set, and calculates evaluation metrics
(MSE, RMSE, R^2) for each model. The results are printed to the console for comparison.
Author: Joseph M.P. 

"""

from pkg_house_prices.utils.helpers import read_joblib
from pkg_house_prices.utils.project_root import PROJECT_ROOT
from pkg_house_prices.utils.config import CONFIG            
from sklearn.metrics import mean_squared_error, r2_score
from pkg_house_prices.data.data_loader import X_test, y_test
from pkg_house_prices.utils.logger import logger
import pandas as pd


# Evaluate and compare models
def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models and compare their performance.

    Parameters
    ----------
    models : dict
        Dictionary of model name to trained model pipeline
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        Test target variable

    Returns
    -------
    results : dict
        Dictionary of model name to evaluation metrics
    """
    results = {}
    for model_name, model in models.items():
        logger.info(f"evaluate_models() - Evaluating model: {model_name} ...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'R^2': r2
        }
        logger.info(f"evaluate_models() - {model_name} - MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
    return results  



if __name__ == "__main__":

    # Load trained models
    output_path = CONFIG['models']['output_path']
    linear_regression_model = read_joblib(PROJECT_ROOT / output_path, "linear_regression.joblib")       
    ridge_regression_model = read_joblib(PROJECT_ROOT / output_path, "ridge_regression.joblib")
    lasso_regression_model = read_joblib(PROJECT_ROOT / output_path, "lasso_regression.joblib")
    elasticnet_regression_model = read_joblib(PROJECT_ROOT / output_path, "elasticnet_regression.joblib")   
    x_gb_model = read_joblib(PROJECT_ROOT / output_path, "xgboost_regression.joblib")   
    models = {
        "Linear Regression": linear_regression_model,        
        "Ridge Regression": ridge_regression_model,
        "Lasso Regression": lasso_regression_model,
        "ElasticNet Regression": elasticnet_regression_model,
        "XGBoost Regression": x_gb_model    
    }  

    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    for model_name, metrics in results.items():
        print(f"{model_name}: MSE={metrics['MSE']:,.2f}, RMSE={metrics['RMSE']:,.2f}, R^2={metrics['R^2']:,.2f}")