from pathlib import Path
import tensorflow as tf
import numpy as np

_PARENT = Path(__file__).parent

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255  # normalize
    x_test = x_test.astype("float32") / 255  # normalize

    # redimensionar a 7x7
    x_train = np.array([tf.image.resize(np.expand_dims(image, axis=-1), (7, 7)).numpy() for image in x_train])
    
    x_test = np.array([tf.image.resize(np.expand_dims(image, axis=-1), (7, 7)).numpy() for image in x_test])

    # flatten input
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                10, activation="softmax", input_shape=(x_train.shape[-1],)
            ),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

    model_dir = _PARENT / "models"
    model_dir.mkdir(exist_ok=True)

    model.save(f"{model_dir}/mnist.h5")

if __name__ == "__main__":
    main()
