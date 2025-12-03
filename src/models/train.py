import tensorflow as tf
from tensorflow.keras import layers, models
def train_model(X_train, y_train):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    model.save('deployment/model_tf.h5')
    return model
