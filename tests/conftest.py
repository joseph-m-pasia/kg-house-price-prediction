from sklearn.datasets import make_regression

import pytest
import pandas as pd


# define the sample dataframe as a fixture
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "YearBuilt": [2000, 1990],
            "YearRemodAdd": [2010, 2005],
            "YrSold": [2020, 2020],
            "TotalBsmtSF": [1000, 800],
            "1stFlrSF": [1200, 1100],
            "2ndFlrSF": [500, 600],
            "FullBath": [2, 1],
            "HalfBath": [1, 0],
            "BsmtFullBath": [1, 1],
            "BsmtHalfBath": [0, 1],
            "GarageArea": [400, 0],
            "PoolArea": [0, 50],
        }
    )


@pytest.fixture
def regression_data():

    X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    y = pd.Series(y)

    return X, y
