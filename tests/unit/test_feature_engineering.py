from pkg_house_prices.features.preprocessor import FeatureEngineer
from tests.unit.conftest import sample_df
import pandas as pd


# 1. Fit returns self
def test_fit_returns_self(sample_df):
    fe = FeatureEngineer()

    assert fe.fit(sample_df) is fe


# 2. HouseAge is computed correctly
def test_house_age_created(sample_df):
    fe = FeatureEngineer()

    result = fe.transform(sample_df)

    expected = sample_df["YrSold"] - sample_df["YearBuilt"]

    assert "HouseAge" in result.columns
    pd.testing.assert_series_equal(
        result["HouseAge"],
        expected,
        check_names=False,
    )


# 3. RemodAge is computed correctly
def test_remod_age_created(sample_df):
    fe = FeatureEngineer()

    result = fe.transform(sample_df)

    expected = sample_df["YrSold"] - sample_df["YearRemodAdd"]

    assert "RemodAge" in result.columns
    pd.testing.assert_series_equal(
        result["RemodAge"],
        expected,
        check_names=False,
    )


# 4. TotalSF is correct
def test_total_sf_created(sample_df):
    fe = FeatureEngineer()

    result = fe.transform(sample_df)

    expected = sample_df["TotalBsmtSF"] + sample_df["1stFlrSF"] + sample_df["2ndFlrSF"]

    assert "TotalSF" in result.columns
    pd.testing.assert_series_equal(
        result["TotalSF"],
        expected,
        check_names=False,
    )


# 5. TotalBathrooms is correct
def test_total_bathrooms_created(sample_df):
    fe = FeatureEngineer()

    result = fe.transform(sample_df)

    expected = (
        sample_df["FullBath"]
        + 0.5 * sample_df["HalfBath"]
        + sample_df["BsmtFullBath"]
        + 0.5 * sample_df["BsmtHalfBath"]
    )

    assert "TotalBathrooms" in result.columns
    pd.testing.assert_series_equal(
        result["TotalBathrooms"],
        expected,
        check_names=False,
    )


# 6. Missing values are filled with zero
def test_missing_numeric_filled():
    df = pd.DataFrame({"TotalBsmtSF": [None], "1stFlrSF": [900], "2ndFlrSF": [0]})

    fe = FeatureEngineer()

    result = fe.transform(df)

    assert result.loc[0, "TotalBsmtSF"] == 0


# 7. HasBasement
def test_has_basement():
    df = pd.DataFrame({"TotalBsmtSF": [0, 500]})

    fe = FeatureEngineer()

    result = fe.transform(df)

    assert result["HasBasement"].tolist() == [0, 1]


# 9. HasPool
def test_has_pool():
    df = pd.DataFrame({"PoolArea": [0, 100]})

    fe = FeatureEngineer()

    result = fe.transform(df)

    assert result["HasPool"].tolist() == [0, 1]


# 11. Original dataframe is unchanged
def test_original_dataframe_not_modified(sample_df):
    original = sample_df.copy(deep=True)

    FeatureEngineer().transform(sample_df)

    pd.testing.assert_frame_equal(sample_df, original)
