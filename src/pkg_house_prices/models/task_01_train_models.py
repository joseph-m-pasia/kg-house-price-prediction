import os
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
    Train a ML model using sklearn Pipeline.
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
        regressor = XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
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
        avg_cv_score = grid_search.best_score_
        std_cv_score = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        train_score = model.score(X_train, y_train)
        logger.info(f"train_model() - Training R^2 score: {train_score:.4f}")
        logger.info(f"train_model() - Best parameters found: {grid_search.best_params_}")
        logger.info(f"train_model() - Best CV R^2 score: {avg_cv_score:.4f}")
    else:
        logger.info("train_model() - No hyperparameter tuning for Linear Regression.")              
        # 1. Calculate CV score first to evaluate generalizability
        cv_scores = cross_val_score(model, X_train, y_train, cv=CONFIG["params"]["cv_size"], scoring='r2')
        avg_cv_score = cv_scores.mean()
        std_cv_score = cv_scores.std()
        train_score = model.fit(X_train, y_train).score(X_train, y_train)
        logger.info(f"train_model() - Training R^2 score: {train_score:.4f}")
        # 2. Fit the model on the ENTIRE training set for the final "production" version
        model.fit(X_train, y_train)        
        logger.info(f"train_model() - CV R^2 score: {avg_cv_score:.4f}")

    logger.info("train_model() - Dumping pipeline...")
    # create output path if it doesn't exist
    output_path = CONFIG['models']['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    joblib.dump(model, f"{output_path}{model_type}_regression.joblib")
    
    return model, avg_cv_score, train_score, std_cv_score   

if __name__ == "__main__":

    # show me the variables in X_train with NaN values
    nan_columns = X_train.columns[X_train.isna().any()].tolist() 
    logger.info(f"Columns in X_train with NaN values: {nan_columns}")

    # Train the model
    lr_model,    lr_cv_score, lr_train_score, lr_std_cv_score = train_model(X_train, y_train, "linear")
    lasso_model, lasso_cv_score, lasso_train_score, lasso_std_cv_score = train_model(X_train, y_train, "lasso")
    ridge_model, ridge_cv_score, ridge_train_score, ridge_std_cv_score = train_model(X_train, y_train, "ridge")
    enet_model,  elasticnet_cv_score, elasticnet_train_score, elasticnet_std_cv_score = train_model(X_train, y_train, "elasticnet")    
    x_gb_model,  x_gb_cv_score, x_gb_train_score, x_gb_std_cv_score = train_model(X_train, y_train, "xgboost")  

    # print CV scores, train scores, and std CV scores for all models and identify champion model
    cv_scores = {
        "Linear Regression": lr_cv_score,
        "Lasso Regression": lasso_cv_score,
        "Ridge Regression": ridge_cv_score,
        "ElasticNet Regression": elasticnet_cv_score,
        "XGBoost Regression": x_gb_cv_score
    }
    train_scores = {
        "Linear Regression": lr_model.score(X_train, y_train) if lr_model else None,
        "Lasso Regression": lasso_train_score,
        "Ridge Regression": ridge_train_score,
        "ElasticNet Regression": elasticnet_train_score,
        "XGBoost Regression": x_gb_train_score
    }       
    std_cv_scores = {
        "Linear Regression": lr_std_cv_score,
        "Lasso Regression": lasso_std_cv_score,
        "Ridge Regression": ridge_std_cv_score,         
        "ElasticNet Regression": elasticnet_std_cv_score,
        "XGBoost Regression": x_gb_std_cv_score
    }
    for model_name, cv_score in cv_scores.items():
        logger.info(f"{model_name} - CV R^2 Score: {cv_score:.4f}")     
        logger.info(f"{model_name} - Train R^2 Score: {train_scores[model_name]:.4f}")
        logger.info(f"{model_name} - CV R^2 Score Std Dev: {std_cv_scores[model_name]:.4f}")    

    champion_model = max(cv_scores, key=cv_scores.get)  
    logger.info(f"Champion model based on CV R^2 score: {champion_model} with score {cv_scores[champion_model]:.4f}, Train R^2 score: {train_scores[champion_model]:.4f}, CV R^2 score std dev: {std_cv_scores[champion_model]:.4f}")   

    # potentially save champion model separately for easy loading in production
    champion_model_mapping = {
        "Linear Regression": lr_model,
        "Lasso Regression": lasso_model,
        "Ridge Regression": ridge_model,
        "ElasticNet Regression": enet_model,
        "XGBoost Regression": x_gb_model
    }
    champion_model_pipeline = champion_model_mapping[champion_model]
    joblib.dump(champion_model_pipeline, CONFIG["models"]["champion_model"])
    logger.info(f"Champion model pipeline saved to: {CONFIG['models']['champion_model']}")  

    # print the champion model's hyperparameters (if applicable)
    if champion_model != "Linear Regression":
        logger.info(f"Champion model hyperparameters: {champion_model_pipeline.named_steps['regressor']}")  