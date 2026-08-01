"""
To execute locally:
1. Run the following command in the terminal: uvicorn app.main:app --reload
2. Open the browser and navigate to http://127.0.0.1:8000/docs # to view the API documentation and test the endpoints.
3. Use the /predict endpoint to send a POST request with the required input data for prediction.
4. The response will include the prediction, probability, and risk category.
"""

import pandas as pd

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pathlib import Path

from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.helpers import load_ml_model

from app.schemas.schemas import PredictionRequest, PredictionResponse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_PATH = Path("artifacts/xgboost_regression.joblib")

model_bundle = None


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

model_bundle = None


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
# Model loading
# -----------------------------------------------------------------------------
def get_model():

    logger.info("Loading model and feature names...")
    global model_bundle
    if model_bundle is None:
        # Load the bundle saved by save_model()
        model_bundle = load_ml_model(MODEL_PATH)

    # Return the model
    return model_bundle


# -----------------------------------------------------------------------------
# Prediction endpoint
# -----------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    """
    Endpoint to make predictions using the trained model.
    Args:
        data (PredictionRequest): Input data for prediction.
    Returns:
        PredictionResponse: The prediction result and probability.
    """

    logger.info("Received prediction request...")

    # load the model and feature names
    model, feature_names = get_model()

    try:

        # Create a DataFrame from the input data
        df = pd.DataFrame([data.model_dump()])

        # Ensure the DataFrame has the same columns as the model was trained on
        if feature_names is not None:
            df = df[feature_names]

        # Predict the outcome
        prediction = model.predict(df)[0]

        return PredictionResponse(
            prediction=float(prediction)
        )

    except Exception as e:
        logger.error(f"Error occurred while making prediction: {e}")
        raise HTTPException(status_code=500, detail="Error occurred while making prediction")
