"""
To execute locally:
1. Run the following command in the terminal: uvicorn app.main:app --reload
2. Open the browser and navigate to http://127.0.0.1:8000/docs # to view the API documentation and test the endpoints.
3. Use the /predict endpoint to send a POST request with the required input data for prediction.
4. The response will include the prediction, probability, and risk category.
"""

import pandas as pd

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path

from pkg_house_prices.utils.logger import logger

from app.schemas.schemas import PredictionRequest, PredictionResponse
from app.model_loader import get_model
from app.schemas.defaults import DEFAULT_FEATURES

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_PATH = Path("artifacts/model.joblib")

# -----------------------------------------------------------------------------
# FastAPI lifespan
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model when the API starts.
    """
    get_model()
    yield


app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# Health endpoint
# -----------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "i am healthy"}


# -----------------------------------------------------------------------------
# Root endpoint
# -----------------------------------------------------------------------------


@app.get("/")
def root():
    return {"message": "House Price Prediction API", "version": "1.0.0", "docs": "/docs", "health": "/health"}


# -----------------------------------------------------------------------------
# Prediction endpoint
# -----------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PredictionRequest) -> PredictionResponse:
    """
    Predict the sale price of a house.

    Parameters
    ----------
    features : PredictionRequest
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

    logger.info("predict() - Input columns:")
    logger.info(X.columns.tolist())
    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    pipeline = get_model()
    prediction = pipeline.predict(X)[0]

    # ---------------------------------------------------------
    # Return response object
    # ---------------------------------------------------------

    return PredictionResponse(predicted_sale_price=float(prediction))
