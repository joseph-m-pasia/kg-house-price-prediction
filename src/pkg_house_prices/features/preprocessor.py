import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from pkg_house_prices.utils.config import CONFIG

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for preprocessing:
    - Imputes missing values
    - Encodes categorical features
    - Scales numerical features
    """
    def __init__(self):
        # We will define pipelines later in fit
        self.preprocessor_ = None
    
    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        X_df = pd.DataFrame(X)
        
        # Automatically detect column types
        self.categorical_cols_ = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numerical_cols_ = X_df.select_dtypes(include=["number"]).columns.tolist()
        
        # Numerical pipeline
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=CONFIG["features"]["simple_imputer_strategy"])),
            ("scaler", StandardScaler())
        ])
        
        # Categorical pipeline
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        # Combine into ColumnTransformer
        self.preprocessor_ = ColumnTransformer([
            ("num", num_pipeline, self.numerical_cols_),
            ("cat", cat_pipeline, self.categorical_cols_)
        ])
        
        # Fit ColumnTransformer
        self.preprocessor_.fit(X_df)
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X)
        return self.preprocessor_.transform(X_df)
