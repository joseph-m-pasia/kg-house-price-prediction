from matplotlib.path import Path

from pkg_house_prices.models.evaluator import evaluate_models
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.models.trainer import train_model_pipeline
from pkg_house_prices.data.data_loader import load_data

import joblib

# ==================== LAUNCH THE TRAINING PIPELINE ==============================    git s

run_train = False
run_evaluate = False
copy_champion = True

# -------------------------------
# Train the model
# -------------------------------
if run_train:

    lr_model, lr_cv_score, lr_train_score, lr_std_cv_score = train_model_pipeline("linear")
    lasso_model, lasso_cv_score, lasso_train_score, lasso_std_cv_score = train_model_pipeline("lasso")
    ridge_model, ridge_cv_score, ridge_train_score, ridge_std_cv_score = train_model_pipeline("ridge")
    enet_model, elasticnet_cv_score, elasticnet_train_score, elasticnet_std_cv_score = train_model_pipeline(
        "elasticnet"
    )
    x_gb_model, x_gb_cv_score, x_gb_train_score, x_gb_std_cv_score = train_model_pipeline("xgboost")

    # -------------------------------
    # print CV scores, train scores, and std CV scores for all models
    # ------------------------------

    cv_scores = {
        "Linear Regression": lr_cv_score,
        "Lasso Regression": lasso_cv_score,
        "Ridge Regression": ridge_cv_score,
        "ElasticNet Regression": elasticnet_cv_score,
        "XGBoost Regression": x_gb_cv_score,
    }

    std_cv_scores = {
        "Linear Regression": lr_std_cv_score,
        "Lasso Regression": lasso_std_cv_score,
        "Ridge Regression": ridge_std_cv_score,
        "ElasticNet Regression": elasticnet_std_cv_score,
        "XGBoost Regression": x_gb_std_cv_score,
    }

    for model_name, cv_score in cv_scores.items():
        logger.info(f"{model_name} - CV R^2 Score: {cv_score:.4f}")
        logger.info(f"{model_name} - CV R^2 Score Std Dev: {std_cv_scores[model_name]:.4f}")

    # -------------------------------
    # Identify the champion model based on CV R^2 score
    # -------------------------------

    champion_model = max(cv_scores, key=cv_scores.get)
    logger.info(
        f"Champion model based on CV R^2 score: {champion_model} with score {cv_scores[champion_model]:.4f}, CV R^2 score std dev: {std_cv_scores[champion_model]:.4f}"
    )

    # print the champion model's hyperparameters (if applicable)
    if champion_model != "Linear Regression":
        champion_model_pipeline = {
            "Lasso Regression": lasso_model,
            "Ridge Regression": ridge_model,
            "ElasticNet Regression": enet_model,
            "XGBoost Regression": x_gb_model,
        }[champion_model]
        logger.info(f"Champion model hyperparameters: {champion_model_pipeline.named_steps['regressor']}")


# -------------------------------
# Evaluate the champion model on the test set
# -------------------------------

if run_evaluate:

    # load test data
    test_data = load_data(data_path="D:/Joseph/Projects/kg-house-price-prediction/data/test_features.csv")

    X_test = test_data.drop(columns="SalePrice")
    y_test = test_data["SalePrice"]

    # load champion model
    model = joblib.load("artifacts/trained_models/xgboost_regression.joblib")

    # evaluate the model using the test data                    ]
    results = evaluate_models({"xgb": model}, X_test, y_test)

    # print the  variable names of the raw input features used in the model
    print(model.named_steps["feature_engineer"])
    print(model.named_steps.keys())
    print(hasattr(model, "feature_names_in_"))
    print(getattr(model, "feature_names_in_", None))

if copy_champion:

    # copy the champion model to the artifacts directory
    import shutil

    champion_model_dir = "artifacts/trained_models/xgboost_regression.joblib"

    shutil.copy(champion_model_dir, "artifacts/model.joblib")
