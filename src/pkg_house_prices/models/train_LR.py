from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.features.build_features import train_final, test_final
from pkg_house_prices.utils.config import CONFIG

def train_model(X_train, y_train):
    """
    Train a Linear Regression model using sklearn Pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Target variable

    Returns
    -------
    model : sklearn Pipeline
        Trained linear regression pipeline
    """

    logger.info("train_model() - Training LinearRegression model: Define Pipeline   ...")
    model = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    logger.info("train_model() - Fitting pipeline...")
    model.fit(X_train, y_train)

    logger.info("train_model() - Dumping pipeline...")
    joblib.dump(model, 'deployment/linear_regression.joblib')

    return model

# Separate features and target variable
target_variable = CONFIG["data"]["target"]  
X_train = train_final.drop(columns=[target_variable])
y_train = train_final[target_variable]

# show me the variables in X_train with NaN values
nan_columns = X_train.columns[X_train.isna().any()].tolist() 
logger.info(f"Columns in X_train with NaN values: {nan_columns}")

# Train the model
linear_regression_model = train_model(X_train, y_train)

