"""
This module performs hyperparameter optimization for the XGBoost regression model using a systematic approach. It first determines the optimal number of trees (n_estimators) using early stopping on a validation set, then tunes complexity parameters (max_depth, min_child_weight), followed by gamma, regularization parameters (reg_alpha, reg_lambda), subsample and colsample_bytree, and finally the learning rate. The optimized model is evaluated using cross-validation R^2 scores before being finalized as the champion model and saved to disk for later use in prediction.
Author: Joseph M.P.

"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor   
import xgboost as xgb   
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.model_selection import KFold


from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.config import CONFIG
from pkg_house_prices.data.data_loader import X_train, y_train
from pkg_house_prices.features.missing_ratio_dropper import MissingRatioDropper
from pkg_house_prices.features.preprocessor import Preprocessor
from pkg_house_prices.features.preprocessor import FeatureEngineer  
from pkg_house_prices.models.task_02_evaluate_models import evaluate_models

#######################################################################################################
#  STEP 1: Determine the number of trees (n_estimators) using early stopping on a validation set
#######################################################################################################

def optimize_nb_trees(X_train, y_train):   
    """
    Train an XGBoost regression model on tabular data with custom preprocessing, using manual KFold cross-validation to determine optimal boosting rounds (trees) via early stopping. The workflow ensures that preprocessing steps, feature engineering, and missing value handling are applied consistently across folds. The final model is trained on the full dataset using the average optimal number of trees.    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable
    
    Returns
    -------
    model : XGBRegressor
        Trained XGBoost regression model with optimal number of trees
    """
    logger.info("optimize_nb_trees() - Starting training with early stopping...")
    
    # Define the preprocessing pipeline
    logger.info("optimize_nb_trees() - Defining preprocessing pipeline...")
    model = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('dropper', MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
        ('preprocessor', Preprocessor())])   
    
    # Preprocess the training data
    X_train_processed = model.fit_transform(X_train, y_train)
    
    # Set up KFold cross-validation
    logger.info("optimize_nb_trees() - Setting up KFold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    params = {
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1,
        "objective": "reg:squarederror",
        "seed": 42,
        "early_stopping_rounds": 100
    }
    
    best_n_estimators = []

    for train_index, val_index in kf.split(X_train_processed):

        X_train_fold, X_val_fold = X_train_processed[train_index], X_train_processed[val_index]
  
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
        xgb_model = XGBRegressor(**params, 
                                 n_estimators=5000)
        xgb_model.fit(X_train_fold, 
                      y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      verbose=False)
        
        best_n_estimators.append(xgb_model.best_iteration+1)  # +1 because best_iteration_ is zero-indexed
    
    avg_best_n_estimators = int(np.mean(best_n_estimators))
    
    logger.info(f"optimize_nb_trees() - Average optimal number of trees: {avg_best_n_estimators}")
        
    return avg_best_n_estimators

def tune_complexity_parameters(X_train, y_train, n_estimators):
    """
    Tune XGBoost complexity parameters (max_depth, min_child_weight) using GridSearchCV with the optimal number of trees determined from early stopping.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable
    n_estimators : int
        Optimal number of trees determined from early stopping
    
    Returns
    -------
    best_params : dict
        Best hyperparameters found from GridSearchCV
    """
    logger.info("tune_complexity_parameters() - Starting hyperparameter tuning...")
    
    model = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('dropper', MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
        ('preprocessor', Preprocessor()),
        ('regressor', XGBRegressor(objective='reg:squarederror', 
                                   eval_metric='rmse', 
                                   n_estimators=
                                   n_estimators, 
                                   seed=42))
    ])
    
    param_grid = {
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_child_weight': [1, 3, 5]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=CONFIG["params"]["cv_size"], scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"tune_complexity_parameters() - Best parameters found: {grid_search.best_params_}")
    
    return grid_search.best_params_

def tune_gamma_parameter(X_train, y_train, n_estimators, max_depth, min_child_weight):
    """
    Tune XGBoost gamma parameter using GridSearchCV with the optimal number of trees and complexity parameters determined from previous steps.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable
    n_estimators : int
        Optimal number of trees determined from early stopping
    max_depth : int
        Optimal max_depth determined from previous tuning step
    min_child_weight : int
        Optimal min_child_weight determined from previous tuning step
    
    Returns
    -------
    best_gamma : float
        Best gamma value found from GridSearchCV
    """
    logger.info("tune_gamma_parameter() - Starting gamma tuning...")
    
    model = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('dropper', MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
        ('preprocessor', Preprocessor()),
        ('regressor', XGBRegressor(objective='reg:squarederror', 
                                   eval_metric='rmse', 
                                   n_estimators=n_estimators, 
                                   max_depth=max_depth, 
                                   min_child_weight=min_child_weight, 
                                   seed=42))
    ])
    
    param_grid = {
        'regressor__gamma': [0, 0.1, 0.2, 0.3]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=CONFIG["params"]["cv_size"], scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"tune_gamma_parameter() - Best gamma found: {grid_search.best_params_['regressor__gamma']}")
    
    return grid_search.best_params_['regressor__gamma']


def tune_regalpha_reglambda(X_train, y_train, n_estimators, max_depth, min_child_weight, gamma):
    """
    Tune XGBoost regularization parameters (reg_alpha, reg_lambda) using GridSearchCV with the optimal number of trees and other parameters determined from previous steps.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable
    n_estimators : int
        Optimal number of trees determined from early stopping
    max_depth : int
        Optimal max_depth determined from previous tuning step
    min_child_weight : int
        Optimal min_child_weight determined from previous tuning step
    gamma : float
        Optimal gamma determined from previous tuning step
    
    Returns
    -------
    best_params : dict
        Best regularization parameters found from GridSearchCV
    """
    logger.info("tune_regalpha_reglambda() - Starting regularization parameter tuning...")
    
    model = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('dropper', MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
        ('preprocessor', Preprocessor()),
        ('regressor', XGBRegressor(objective='reg:squarederror', 
                                   eval_metric='rmse', 
                                   n_estimators=n_estimators, 
                                   max_depth=max_depth, 
                                   min_child_weight=min_child_weight, 
                                   gamma=gamma,
                                   seed=42))
    ])
    
    param_grid = {
        'regressor__reg_alpha': [0, 0.1, 0.5, 1],
        'regressor__reg_lambda': [0.5, 1, 1.5, 2]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=CONFIG["params"]["cv_size"], scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"tune_regalpha_reglambda() - Best regularization parameters found: {grid_search.best_params_}")
    
    return grid_search.best_params_

def tune_subsample_colsample(X_train, y_train, n_estimators, max_depth, min_child_weight, gamma, reg_alpha, reg_lambda):
    """
    Tune XGBoost subsample and colsample_bytree parameters using GridSearchCV with the optimal number of trees and other parameters determined from previous steps.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable
    n_estimators : int
        Optimal number of trees determined from early stopping
    max_depth : int
        Optimal max_depth determined from previous tuning step
    min_child_weight : int
        Optimal min_child_weight determined from previous tuning step
    gamma : float
        Optimal gamma determined from previous tuning step
    reg_alpha : float
        Optimal reg_alpha determined from previous tuning step
    reg_lambda : float
        Optimal reg_lambda determined from previous tuning step
    
    Returns
    -------
    best_params : dict
        Best subsample and colsample_bytree parameters found from GridSearchCV
    """
    logger.info("tune_subsample_colsample() - Starting subsample and colsample tuning...")
    
    model = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('dropper', MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
        ('preprocessor', Preprocessor()),
        ('regressor', XGBRegressor(objective='reg:squarederror', 
                                   eval_metric='rmse', 
                                   n_estimators=n_estimators, 
                                   max_depth=max_depth, 
                                   min_child_weight=min_child_weight, 
                                   gamma=gamma,
                                   reg_alpha=reg_alpha,
                                   reg_lambda=reg_lambda,
                                   seed=42))
    ])
    
    param_grid = {
        'regressor__subsample': [0.6, 0.8, 1.0],
        'regressor__colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=CONFIG["params"]["cv_size"], scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"tune_subsample_colsample() - Best subsample and colsample parameters found: {grid_search.best_params_}")
    
    return grid_search.best_params_ 
  
def tune_learning_rate(X_train, y_train, n_estimators, max_depth, min_child_weight, gamma, reg_alpha, reg_lambda, subsample, colsample_bytree):
    """
    Tune XGBoost learning rate using GridSearchCV with the optimal number of trees and other parameters determined from previous steps.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable
    n_estimators : int
        Optimal number of trees determined from early stopping
    max_depth : int
        Optimal max_depth determined from previous tuning step
    min_child_weight : int
        Optimal min_child_weight determined from previous tuning step
    gamma : float
        Optimal gamma determined from previous tuning step
    reg_alpha : float
        Optimal reg_alpha determined from previous tuning step
    reg_lambda : float
        Optimal reg_lambda determined from previous tuning step
    subsample : float
        Optimal subsample determined from previous tuning step
    colsample_bytree : float
        Optimal colsample_bytree determined from previous tuning step
    
    Returns
    -------
    best_learning_rate : float
        Best learning rate found from GridSearchCV
    """
    logger.info("tune_learning_rate() - Starting learning rate tuning...")
    
    model = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('dropper', MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
        ('preprocessor', Preprocessor()),
        ('regressor', XGBRegressor(objective='reg:squarederror', 
                                   eval_metric='rmse', 
                                   n_estimators=n_estimators, 
                                   max_depth=max_depth, 
                                   min_child_weight=min_child_weight, 
                                   gamma=gamma,
                                   reg_alpha=reg_alpha,
                                   reg_lambda=reg_lambda,
                                   subsample=subsample,
                                   colsample_bytree=colsample_bytree,
                                   seed=42))
                            ])
    
    param_grid = {
        'regressor__learning_rate': [0.01, 0.05, 0.1]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=CONFIG["params"]["cv_size"], scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"tune_learning_rate() - Best learning rate found: {grid_search.best_params_['regressor__learning_rate']}")
    
    return grid_search.best_params_['regressor__learning_rate']
    

if __name__ == "__main__":
    avg_n_estimators         = optimize_nb_trees(X_train, y_train)
    
    best_complexity_params   = tune_complexity_parameters(X_train, y_train, avg_n_estimators)
    
    best_gamma               = tune_gamma_parameter(X_train, y_train, avg_n_estimators, best_complexity_params        ['regressor__max_depth'], best_complexity_params['regressor__min_child_weight'])
    
    best_reg_params          = tune_regalpha_reglambda(X_train, y_train, avg_n_estimators, best_complexity_params['regressor__max_depth'], best_complexity_params['regressor__min_child_weight'], best_gamma)
    
    best_subsample_colsample = tune_subsample_colsample(X_train, y_train, avg_n_estimators, best_complexity_params['regressor__max_depth'], best_complexity_params['regressor__min_child_weight'], best_gamma, best_reg_params['regressor__reg_alpha'], best_reg_params['regressor__reg_lambda'])
    
    best_learning_rate       = tune_learning_rate(X_train, y_train, avg_n_estimators, best_complexity_params['regressor__max_depth'], best_complexity_params['regressor__min_child_weight'], best_gamma, best_reg_params['regressor__reg_alpha'], best_reg_params['regressor__reg_lambda'], best_subsample_colsample['regressor__subsample'], best_subsample_colsample['regressor__colsample_bytree'])    

    # Evaluate final model with best hyperparameters on a validation set or through cross-validation to confirm performance improvements before finalizing the champion model.
    model = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('dropper', MissingRatioDropper(min_ratio=CONFIG["params"]["missing_threshold"])),
        ('preprocessor', Preprocessor()),
        ('regressor', XGBRegressor(objective='reg:squarederror', 
                                        eval_metric='rmse',
                                        n_estimators=avg_n_estimators,
                                        max_depth=best_complexity_params['regressor__max_depth'],                   
                                        min_child_weight=best_complexity_params['regressor__min_child_weight'],
                                        gamma=best_gamma,
                                        reg_alpha=best_reg_params['regressor__reg_alpha'],                          
                                        reg_lambda=best_reg_params['regressor__reg_lambda'],
                                        subsample=best_subsample_colsample['regressor__subsample'],
                                        colsample_bytree=best_subsample_colsample['regressor__colsample_bytree'],
                                        learning_rate=best_learning_rate,
                                        seed=42))
                            ])
    

    # Train using CV to confirm performance improvements before finalizing the champion model
    cv_scores = cross_val_score(model, X_train, y_train, cv=CONFIG["params"]["cv_size"], scoring='r2', n_jobs=-1)
    logger.info(f"Final model CV R^2 score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}") 

    # Train using full training data with the best hyperparameters found
    model.fit(X_train, y_train)
    evaluate_models({"Optimized XGBoost": model}, X_train, y_train)

    # save the final model
    output_path = CONFIG['models']['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    joblib.dump(model, f"{output_path}champion_xgboost_regression.joblib")     
    

