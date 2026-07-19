from sklearn.datasets import make_regression

import pytest
import pandas as pd


# define the sample dataframe as a fixture
@pytest.fixture
def sample_df():

    data_path = "tests/sample_data/train_sample.csv"
    dt = pd.read_csv(data_path)

    # exclude ID and SalePrice columns for features, and use SalePrice as target
    X = dt.drop(columns=["SalePrice"])
    y = dt["SalePrice"]
    return X, y


@pytest.fixture
def regression_data(sample_df):

    X, y = sample_df

    # X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)

    # X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # y = pd.Series(y)

    return X, y
