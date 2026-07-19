"""
This module evaluates the performance of trained regression models (Linear, Ridge, Lasso, ElasticNet, XGBoost) on the test set.
It loads the trained models from disk, makes predictions on the test set, and calculates evaluation metrics
(MSE, RMSE, R^2) for each model. The results are printed to the console for comparison.
Author: Joseph M.P.

"""

from pkg_house_prices.utils.helpers import read_joblib
from pkg_house_prices.utils.project_root import PROJECT_ROOT
from pkg_house_prices.utils.config import CONFIG
from pkg_house_prices.utils.logger import logger

from sklearn.metrics import mean_squared_error, r2_score


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
        rmse = mse**0.5
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {"MSE": mse, "RMSE": rmse, "R^2": r2}
        logger.info(f"evaluate_models() - {model_name} - MSE: {mse}, RMSE: {rmse}, R^2: {r2}")

    return results
