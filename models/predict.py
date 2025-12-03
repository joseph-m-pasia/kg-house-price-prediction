import tensorflow as tf
def predict(model_path, X):
    model = tf.keras.models.load_model(model_path)
    return model.predict(X)
