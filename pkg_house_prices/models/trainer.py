"""
This module trains multiple regression models (Linear, Ridge, Lasso, ElasticNet, XGBoost) using sklearn Pipelines.
It performs hyperparameter tuning using GridSearchCV for Ridge, Lasso, ElasticNet, and XGBoost, and evaluates model performance using cross-validation R^2 scores.
The trained models are saved to disk using joblib for later use in prediction.
Author: Joseph M.P.

"""

import os
import joblib

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.config import CONFIG
from pkg_house_prices.features.missing_ratio_dropper import MissingRatioDropper
from pkg_house_prices.features.preprocessor import Preprocessor
from pkg_house_prices.features.preprocessor import FeatureEngineer
from pkg_house_prices.data.data_loader import load_data, extract_features_target, split_data, save_data


def data_pipeline(
    data_path="",
    target_variable="Y",
    test_size=0.2,
    random_state=42,
    transform_y=False,
    save_to="",
    save_as_train="train_features.csv",
    save_as_test="test_features.csv",
):
    """
    Load data, split into features and target variable, and split into training and testing sets.
    Returns X_train, X_test, y_train, y_test
    """
    dt = load_data(data_path)
    X, y = extract_features_target(dt, target_variable)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, transform_y=transform_y
    )
    save_data(X_train, y_train, save_as=save_as_train, save_to=save_to)
    save_data(X_test, y_test, save_as=save_as_test, save_to=save_to)

    return X_train, X_test, y_train, y_test


def train_model_pipeline(model_type="linear", data_path = CONFIG["data"]["train"]):
    """
    Train a ML model using sklearn Pipeline.
    Supports Linear, Ridge, Lasso, and ElasticNet regression.

    Parameter
    ----------
    model_type : str
        Type of regression model to train: 'linear', 'ridge', 'lasso', 'elasticnet', 'xgboost'
    Returns
    -------
    model : sklearn Pipeline
        Trained linear regression pipeline
    """

    # ------------------------
    # Load Data
    # -----------------------

    X_train, X_test, y_train, y_test = data_pipeline(
        data_path=data_path,
        target_variable=CONFIG["data"]["target"],
        test_size=CONFIG["training"]["test_size"],
        random_state=CONFIG["training"]["random_seed"],
        transform_y=CONFIG["data"]["transform_y"],
        save_to=CONFIG["data"]["final_output_path"],
        save_as_train="train_features.csv",
        save_as_test="test_features.csv",
    )

    # -----------------------
    # Model Setup Before Training
    # -----------------------

    model_type = model_type.lower()
    logger.info(f"train_model() - Training model of type: {model_type} ...")
    if model_type == "linear":
        regressor = LinearRegression()
        param_grid = {}
    elif model_type == "ridge":
        regressor = Ridge()
        param_grid = {"regressor__alpha": CONFIG["params"]["regressor_alpha"]}
    elif model_type == "lasso":
        regressor = Lasso(max_iter=5000, tol=1e-5)
        param_grid = {"regressor__alpha": CONFIG["params"]["regressor_alpha"]}
    elif model_type == "elasticnet":
        regressor = ElasticNet(max_iter=5000, tol=1e-5)
        param_grid = {
            "regressor__alpha": CONFIG["params"]["regressor_alpha"],
            "regressor__l1_ratio": CONFIG["params"]["regressor__l1_ratio"],
        }
    elif model_type == "xgboost":
        regressor = XGBRegressor(objective="reg:squarederror", eval_metric="rmse")
        param_grid = {
            "regressor__n_estimators": CONFIG["params"]["xgb_n_estimators"],
            "regressor__max_depth": CONFIG["params"]["xgb_max_depth"],
            "regressor__learning_rate": CONFIG["params"]["xgb_learning_rate"],
        }
    else:
        raise ValueError("model_type must be one of 'linear', 'ridge', 'lasso', 'elasticnet'")

    # -----------------------
    # Pipeline Setup
    # -----------------------

    logger.info("train_model() - Fitting pipeline...")

    model = Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("dropper", MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
            ("preprocessor", Preprocessor()),
            ("regressor", regressor),
        ]
    )

    # -----------------------
    # Model Training Using GridSearchCV for Hyperparameter Tuning
    # -----------------------

    if param_grid:
        logger.info("train_model() - Performing GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(model, param_grid, cv=CONFIG["params"]["cv_size"], scoring="r2", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        avg_cv_score = grid_search.best_score_
        std_cv_score = grid_search.cv_results_["std_test_score"][grid_search.best_index_]
        train_score = model.score(X_train, y_train)
        logger.info(f"train_model() - Training R^2 score: {train_score:.4f}")
        logger.info(f"train_model() - Best parameters found: {grid_search.best_params_}")
        logger.info(f"train_model() - Best CV R^2 score: {avg_cv_score:.4f}")
    else:
        logger.info("train_model() - No hyperparameter tuning for Linear Regression.")
        # 1. Calculate CV score first to evaluate generalizability
        cv_scores = cross_val_score(model, X_train, y_train, cv=CONFIG["params"]["cv_size"], scoring="r2")
        avg_cv_score = cv_scores.mean()
        std_cv_score = cv_scores.std()
        train_score = model.fit(X_train, y_train).score(X_train, y_train)
        logger.info(f"train_model() - Training R^2 score: {train_score:.4f}")
        # 2. Fit the model on the ENTIRE training set for the final "production" version
        model.fit(X_train, y_train)
        logger.info(f"train_model() - CV R^2 score: {avg_cv_score:.4f}")

    # -----------------------
    # Save the trained model to disk
    # -----------------------

    logger.info("train_model() - Dumping pipeline...")
    # create output path if it doesn't exist
    output_path = CONFIG["models"]["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    joblib.dump(model, f"{output_path}{model_type}_regression.joblib")

    return model, avg_cv_score, train_score, std_cv_score
