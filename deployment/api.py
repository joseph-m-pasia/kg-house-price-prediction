from fastapi import FastAPI
import tensorflow as tf
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("deployment/model_tf.h5")


@app.post("/predict")
def predict(data: dict):
    X = np.array([list(data.values())])
    pred = model.predict(X)
    return {"prediction": pred.tolist()}
