from unittest.mock import Mock, patch
import numpy as np
import pytest

from app.main import predict
from app.schemas.schemas import PredictionRequest


def test_predict_returns_prediction():

    house = PredictionRequest(
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

    fake_model = Mock()
    fake_model.predict.return_value = [np.log1p(250000)]

    with patch(
        "app.main.get_model",
        return_value=fake_model,
    ):
        result = predict(house)

    assert result.predicted_sale_price == pytest.approx(250000.0, rel=1e-6)
