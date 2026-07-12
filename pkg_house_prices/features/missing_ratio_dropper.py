import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingRatioDropper(BaseEstimator, TransformerMixin):
    def __init__(self, min_ratio=0.8):
        """
        Parameters
        ----------
        min_ratio : float
            Minimum required ratio of non-missing values to keep a column.
            Columns with ratio < min_ratio will be dropped.
        """
        self.min_ratio = min_ratio

    def fit(self, X, y=None):
        # Convert to DataFrame if needed
        X_df = pd.DataFrame(X)

        n_samples = len(X_df)

        # Ratio of non-missing values per column
        self.non_missing_ratio_ = 1 - X_df.isna().sum() / n_samples

        # Columns to keep
        self.columns_to_keep_ = self.non_missing_ratio_[self.non_missing_ratio_ >= self.min_ratio].index.tolist()

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)

        return X_df[self.columns_to_keep_]
