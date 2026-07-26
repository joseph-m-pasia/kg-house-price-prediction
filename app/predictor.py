"""
predictor.py

Prediction engine for the House Price Prediction application.
"""

from __future__ import annotations

import pandas as pd

from app.model_loader import get_pipeline
from app.schemas.defaults import DEFAULT_FEATURES
from app.schemas.schemas import HouseFeatures, PredictionResponse


def predict(features: HouseFeatures) -> PredictionResponse:
    """
    Predict the sale price of a house.

    Parameters
    ----------
    features : HouseFeatures
        User supplied house features.

    Returns
    -------
    PredictionResponse
        Predicted sale price.
    """

    # ---------------------------------------------------------
    # Convert Pydantic model to dictionary
    # ---------------------------------------------------------

    user_features = features.model_dump(by_alias=True, exclude_none=True)

    # ---------------------------------------------------------
    # Merge defaults with user inputs
    #
    # User values overwrite defaults.
    # ---------------------------------------------------------

    model_input = {**DEFAULT_FEATURES, **user_features}

    # ---------------------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------------------

    X = pd.DataFrame([model_input])

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    pipeline = get_pipeline()
    prediction = pipeline.predict(X)[0]

    # ---------------------------------------------------------
    # Return response object
    # ---------------------------------------------------------

    return PredictionResponse(predicted_sale_price=float(prediction))
