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
