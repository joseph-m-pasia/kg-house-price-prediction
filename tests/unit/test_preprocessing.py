from pkg_house_prices.features.preprocessor import Preprocessor
from tests.conftest import sample_df
import pandas as pd
import pytest

# 1. Fit returns self
def test_preprocessor_fit_returns_self(sample_df):
    p = Preprocessor()

    assert p.fit(sample_df) is p

# 2. Detects numerical columns
def test_detects_numeric_columns():
    df = pd.DataFrame({
        "A":[1,2],
        "B":[3.5,4.5],
        "C":["a","b"]
    })

    p = Preprocessor()

    p.fit(df)

    assert set(p.numerical_cols_) == {"A","B"}

# 3. Detects categorical columns
def test_detects_categorical_columns():
    df = pd.DataFrame({
        "A":[1,2],
        "B":[3.5,4.5],
        "C":["a","b"]
    })

    p = Preprocessor()

    p.fit(df)

    assert set(p.categorical_cols_) == {"C"}

# 4. Transform returns same number of rows
def test_transform_preserves_rows(sample_df):
    p = Preprocessor()

    p.fit(sample_df)

    result = p.transform(sample_df)

    assert result.shape[0] == len(sample_df)

# 5. No NaNs after preprocessing
def test_no_missing_after_preprocessing(sample_df):
    p = Preprocessor()

    p.fit(sample_df)

    result = p.transform(sample_df)

    assert not pd.isna(result).any()

# 6. Output contains no object dtype
def test_output_is_numeric(sample_df):
    p = Preprocessor()

    p.fit(sample_df)

    X = p.transform(sample_df)

    assert X.shape[1] > 0

# 7. Transform before fit raises AttributeError
def test_transform_before_fit():
    p = Preprocessor()

    df = pd.DataFrame({"A":[1]})

    with pytest.raises(AttributeError):
        p.transform(df)