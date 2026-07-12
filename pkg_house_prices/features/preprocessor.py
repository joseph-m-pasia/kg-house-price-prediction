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



class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for House Prices dataset.
    Adds:
      - HouseAge
      - RemodAge
      - TotalSF
      - TotalBathrooms
      - HasBasement
      - HasFireplace
      - HasGarage
      - HasPool
      - HasSecondFloor
    """
    def __init__(self):
        pass  # No parameters for now

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        
        # Numeric features with missing values filled
        for col in ["TotalBsmtSF", "Fireplaces", "GarageCars", "PoolArea", "2ndFlrSF", 
                    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
                    "1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "WoodDeckSF"]:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        
        # Age features
        if all(c in X.columns for c in ["YrSold", "YearBuilt"]):
            X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        if all(c in X.columns for c in ["YrSold", "YearRemodAdd"]):
            X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        
        # Total and derived size features
        size_cols = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
        if all(c in X.columns for c in size_cols):
            X["TotalSF"] = X[size_cols].sum(axis=1)
        if all(c in X.columns for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]):
            X["TotalBathrooms"] = X["FullBath"] + 0.5*X["HalfBath"] + X["BsmtFullBath"] + 0.5*X["BsmtHalfBath"]
        
        # Binary presence features
        if "TotalBsmtSF" in X.columns:
            X["HasBasement"] = (X["TotalBsmtSF"] > 0).astype(int)
        if "Fireplaces" in X.columns:
            X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        if "GarageCars" in X.columns:
            X["HasGarage"] = (X["GarageCars"] > 0).astype(int)
        if "PoolArea" in X.columns:
            X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        if "2ndFlrSF" in X.columns:
            X["HasSecondFloor"] = (X["2ndFlrSF"] > 0).astype(int)
        if "WoodDeckSF" in X.columns:
            X["HasDeck"] = (X["WoodDeckSF"] > 0).astype(int)
        if "Fence" in X.columns:
            X["HasFence"] = X["Fence"].notna().astype(int)
        
        return X
