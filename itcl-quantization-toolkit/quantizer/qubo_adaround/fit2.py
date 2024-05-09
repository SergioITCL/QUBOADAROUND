from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split



_PARENT = Path(__file__).parent
def main():

    X = np.random.uniform(-1, 1, size=(10000,1))
    y = 2*X**3+X**2-X+ 0.1 * np.random.randn(10000, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = tf.keras.Sequential([
        #tf.keras.layers.Dense(units=2**4, activation="tanh"),
        #tf.keras.layers.Dense(units=2**5, activation="tanh"),
        #tf.keras.layers.Dense(units=2**6, activation="tanh"),
        #tf.keras.layers.Dense(units=2**7, activation="tanh"),
        #tf.keras.layers.Dense(units=2**7, activation="tanh"),
        #tf.keras.layers.Dense(units=2**6, activation="tanh"),
        #tf.keras.layers.Dense(units=2**5, activation="tanh"),
        tf.keras.layers.Dense(units=2**4, activation="tanh"),
        tf.keras.layers.Dense(units=2**3, activation="tanh"),
        tf.keras.layers.Dense(units=2**2, activation="tanh"),
        tf.keras.layers.Dense(1)                  # Output layer with a single neuron (for regression)
    ])


    model.compile(optimizer='adam', loss='mean_squared_error')
    # You can adjust the number of epochs and batch size based on your data and resources.
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    model_dir = _PARENT / "models"
    model_dir.mkdir(exist_ok=True)
    np.savetxt('x.txt',X, delimiter="," )
    np.savetxt('y.txt',y, delimiter=",")
    model.save(f"{model_dir}/mnist.h5")
    




if __name__ == "__main__":
    main()