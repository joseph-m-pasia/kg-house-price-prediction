# perform several diagnostics on the trained model
# such as checking for multicollinearity, residual analysis, etc. 
       
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pkg_house_prices.utils.config import CONFIG
from pkg_house_prices.utils.project_root import PROJECT_ROOT
from pkg_house_prices.data.data_loader import X_train, y_train
from pkg_house_prices.utils.helpers import read_joblib
from pkg_house_prices.utils.logger import logger

def perform_diagnostics(model, X_train, y_train):
    """
    Perform diagnostics on the trained model.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained model pipeline
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable

    Returns
    -------
    diagnostics_report : dict
        A report containing various diagnostics results
    """
    logger.info("perform_diagnostics() - Performing model diagnostics...")
    diagnostics_report = {}

    # Understand fit quality using R-squared, MSE, and RMSE
    logger.info("perform_diagnostics() - Calculating R-squared...")
    r_squared = model.score(X_train, y_train)
    logger.info(f"perform_diagnostics() - R-squared on training data: {r_squared}")

    y_pred = model.predict(X_train)
    logger.info("perform_diagnostics() - Calculating MSE and RMSE...")
    mse = mean_squared_error(y_train, y_pred)
    logger.info(f"perform_diagnostics() - MSE on training data: {mse}")
    rmse = mse ** 0.5   
    logger.info(f"perform_diagnostics() - RMSE on training data: {rmse}")
   
    diagnostics_report['r_squared'] = r_squared
    diagnostics_report['mse'] = mse
    diagnostics_report['rmse'] = rmse
 
    # Are error roughly random, not patterned?  
    residuals = y_train - y_pred
    diagnostics_report['residuals'] = residuals
    logger.info("perform_diagnostics() - Residuals calculated.")

    # Visual check: Residuals vs Fitted values plot
    logger.info("perform_diagnostics() - Plotting Residuals vs Fitted Values...")
    fig_bar, ax_bar = plt.subplots(figsize=(8,6))
    ax_bar.scatter(y_pred, residuals, alpha=0.5)
    ax_bar.axhline(y=0, color='r', linestyle='--')
    ax_bar.set_xlabel('Fitted values')
    ax_bar.set_ylabel('Residuals')
    ax_bar.set_title('Residuals vs Fitted Values')

    logger.info("perform_diagnostics() - QQ-plot.")
    # Visual check: QQ-plot for residuals
    fig_qq, ax_qq = plt.subplots(figsize=(8,6))   
    ax_qq.set_title('QQ-Plot of Residuals')
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Sample Quantiles')
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.sort(np.random.normal(0, np.std(residuals), len(residuals)))
    ax_qq.scatter(theoretical_quantiles, sorted_residuals, alpha=0.5)
    ax_qq.plot(theoretical_quantiles, theoretical_quantiles, color='r', linestyle='--')

    # Save the plot to a file in the data output directory if configured 
    if CONFIG["diagnostics"]["to_plot"]:
        logger.info("perform_diagnostics() - Save Displayed Residuals vs Fitted Values plot.")
        try:
            output_dir = PROJECT_ROOT / CONFIG["diagnostics"]["diagnostics_output_path"]
            output_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
            logger.info(f"perform_diagnostics() - Output directory for diagnostics: {output_dir}")
        except Exception as e:
            logger.error(f"perform_diagnostics() - Error creating output directories: {e}") 
        fig_bar.savefig(output_dir / 'residuals_vs_fitted.png', dpi=300)
        fig_qq.savefig(output_dir / 'qq_plot_residuals.png', dpi=300)
        logger.info("perform_diagnostics() - Diagnostic plots saved successfully.")
    
    # Close properly
    plt.close(fig_bar)
    plt.close(fig_qq)

    return diagnostics_report


if __name__ == "__main__":

    # Load trained models
    linear_regression_model = read_joblib(PROJECT_ROOT / "src/pkg_house_prices/data/outputs", "linear_regression.joblib")       
    ridge_regression_model = read_joblib(PROJECT_ROOT / "src/pkg_house_prices/data/outputs", "ridge_regression.joblib")
    lasso_regression_model = read_joblib(PROJECT_ROOT / "src/pkg_house_prices/data/outputs", "lasso_regression.joblib")
    elasticnet_regression_model = read_joblib(PROJECT_ROOT / "src/pkg_house_prices/data/outputs", "elasticnet_regression.joblib")   
    x_gb_model = read_joblib(PROJECT_ROOT / "src/pkg_house_prices/data/outputs", "xgboost_regression.joblib")
    perform_diagnostics(x_gb_model, X_train, y_train)