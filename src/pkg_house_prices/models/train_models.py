from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor   
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import joblib

from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.config import CONFIG
from pkg_house_prices.data.data_loader import X_train, y_train
from pkg_house_prices.features.missing_ratio_dropper import MissingRatioDropper
from pkg_house_prices.features.preprocessor import Preprocessor
from pkg_house_prices.features.preprocessor import FeatureEngineer


def train_model(X_train, y_train, model_type='linear'):
    """
    Train a Linear Regression model using sklearn Pipeline.
    Supports Linear, Ridge, Lasso, and ElasticNet regression.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable
    model_type : str
        Type of regression model to train: 'linear', 'ridge', 'lasso', 'elasticnet', 'xgboost'
    Returns
    -------
    model : sklearn Pipeline
        Trained linear regression pipeline
    """
    model_type = model_type.lower()
    logger.info(f"train_model() - Training model of type: {model_type} ...")         
    if model_type == 'linear':      
        regressor = LinearRegression()
        param_grid = {}
    elif model_type == 'ridge':
        regressor = Ridge()
        param_grid = {'regressor__alpha': CONFIG["params"]["regressor_alpha"]}
    elif model_type == 'lasso':
        regressor = Lasso(max_iter=5000, tol=1e-5)
        param_grid = {'regressor__alpha': CONFIG["params"]["regressor_alpha"]}
    elif model_type == 'elasticnet':
        regressor = ElasticNet(max_iter=5000, tol=1e-5)
        param_grid = {'regressor__alpha': CONFIG["params"]["regressor_alpha"],
                      'regressor__l1_ratio': CONFIG["params"]["regressor__l1_ratio"]}
    elif model_type == 'xgboost':
        regressor = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', use_label_encoder=False)
        param_grid = {
            'regressor__n_estimators': CONFIG["params"]["xgb_n_estimators"],
            'regressor__max_depth': CONFIG["params"]["xgb_max_depth"],
            'regressor__learning_rate': CONFIG["params"]["xgb_learning_rate"]
        }
    else:
         raise ValueError("model_type must be one of 'linear', 'ridge', 'lasso', 'elasticnet'")   

    logger.info("train_model() - Training LinearRegression model: Define Pipeline   ...")
    model = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('dropper', MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
        ('preprocessor', Preprocessor()),
        ('regressor', regressor)
    ])
    logger.info("train_model() - Fitting pipeline...")

    if param_grid:
        logger.info("train_model() - Performing GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(model, param_grid, cv=CONFIG["params"]["cv_size"], scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logger.info(f"train_model() - Best parameters found: {grid_search.best_params_}")
    else:
        logger.info("train_model() - No hyperparameter tuning for Linear Regression.")              
        model.fit(X_train, y_train)
        best_score = cross_val_score(model, X_train, y_train, cv=CONFIG["params"]["cv_size"], scoring='r2').mean()
        logger.info(f"train_model() - CV R^2 score: {best_score:.4f}")

    logger.info("train_model() - Dumping pipeline...")
    joblib.dump(model, f"src/pkg_house_prices/data/outputs/{model_type}_regression.joblib")
    
    return model

if __name__ == "__main__":

    # show me the variables in X_train with NaN values
    nan_columns = X_train.columns[X_train.isna().any()].tolist() 
    logger.info(f"Columns in X_train with NaN values: {nan_columns}")


    # Train the model
    lr_model = train_model(X_train, y_train, "linear")
    lasso_model = train_model(X_train, y_train, "lasso")
    ridge_model = train_model(X_train, y_train, "ridge")
    elasticnet_model = train_model(X_train, y_train, "elasticnet")    
    x_gb_model = train_model(X_train, y_train, "xgboost")  