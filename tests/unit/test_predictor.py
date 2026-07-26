from unittest.mock import Mock, patch

from app.predictor import predict
from app.schemas.schemas import HouseFeatures


def test_predict_returns_prediction():

    house = HouseFeatures(
        OverallQual=7,
        OverallCond=5,
        TotalBsmtSF=900,
        **{
            "1stFlrSF": 900,
            "2ndFlrSF": 700,
        },
        GrLivArea=1600,
        FullBath=2,
        HalfBath=1,
        BsmtFullBath=1,
        BsmtHalfBath=0,
        KitchenQual="Gd",
        KitchenAbvGr=1,
        GarageCars=2,
        GarageQual="TA",
        GarageFinish="Fin",
        GarageType="Attchd",
        BsmtQual="Gd",
        Fireplaces=1,
        CentralAir="Y",
        MSZoning="RL",
        LotShape="Reg",
        PavedDrive="Y",
        YearBuilt=2000,
        YearRemodAdd=2005,
        YrSold=2010,
    )

    fake_pipeline = Mock()
    fake_pipeline.predict.return_value = [250000]

    with patch(
        "app.predictor.pipeline",
        fake_pipeline,
    ):
        result = predict(house)

    assert result.predicted_sale_price == 250000.0