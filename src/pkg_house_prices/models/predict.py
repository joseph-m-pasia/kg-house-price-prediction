from sklearn.linear_model import LinearRegression
import joblib
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.features.build_features import test_final
from pkg_house_prices.utils.config import CONFIG
from pkg_house_prices.models.train_models import linear_regression_model

def predict_house_prices(X):
    """
    Predict house prices using the trained Linear Regression model.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Features for prediction

    Returns
    -------
    predictions : np.ndarray
        Predicted house prices
    """
    logger.info("predict_house_prices() - Making predictions...")
    predictions = linear_regression_model.predict(X)
    logger.info("predict_house_prices() - Predictions completed.")
    return predictions

# Example usage (can be removed in production)
if __name__ == "__main__":
    target_variable = CONFIG["data"]["target"]
    X_test = test_final.drop(columns=[target_variable], errors='ignore')
    preds = predict_house_prices(X_test)
    logger.info(f"Predictions: {preds}")    
    