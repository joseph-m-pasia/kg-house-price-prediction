from pkg_house_prices.utils.logger import logger
from pkg_house_prices.models.trainer import train_model_pipeline

# ==================== LAUNCH THE TRAINING PIPELINE ==============================    git s

# -------------------------------
# Train the model
# -------------------------------

lr_model, lr_cv_score, lr_train_score, lr_std_cv_score = train_model_pipeline("linear")
lasso_model, lasso_cv_score, lasso_train_score, lasso_std_cv_score = train_model_pipeline("lasso")
ridge_model, ridge_cv_score, ridge_train_score, ridge_std_cv_score = train_model_pipeline("ridge")
enet_model, elasticnet_cv_score, elasticnet_train_score, elasticnet_std_cv_score = train_model_pipeline("elasticnet")
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


# X_test, y_test = load_model_data(type="test")

# results = evaluate_models(x_gb_model, X_test, y_test)
# for model_name, metrics in results.items():
#    print(f"{model_name}: MSE={metrics['MSE']:,.2f}, RMSE={metrics['RMSE']:,.2f}, R^2={metrics['R^2']:,.2f}")


# ==================== COPY AND RENAME THE CHAMPION MODEL ==============================
# import shutil

# champion_model_dir = "artifacts/models/20260704_184917_champion_logistic_regression/"


# shutil.copy(f"{champion_model_dir}/champion_logistic_regression_model.pkl", "artifacts/model.pkl")
