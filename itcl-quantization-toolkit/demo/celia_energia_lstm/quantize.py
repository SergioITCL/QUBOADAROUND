import random
from pathlib import Path

import numpy as np
import plotly.express as px
import tensorflow as tf
from itcl_quantizer.tensor_extractor.keras.keras_builder import build as keras_build

BASE = Path(__file__).parent

MODEL = str(BASE / "model/model.h5")


def main():

    X = np.load(str(BASE / "data/inputs.npy"), allow_pickle=False)

    y = np.load(str(BASE / "data/targets.npy"), allow_pickle=True).astype(np.float32)

    SPLIT_IDX = 130219  # from MLFLOW original model split

    X_train, X_test = X[:SPLIT_IDX], X[SPLIT_IDX:]
    y_train, y_test = y[:SPLIT_IDX], y[SPLIT_IDX:]

    rep_dataset_idx = set()

    while len(rep_dataset_idx) < 200:
        rep_dataset_idx.add(random.randint(0, len(X_train) - 1))

    net = keras_build(
        MODEL,
        str(BASE / "model.json"),
        X_train[list(rep_dataset_idx)],
    ).as_sequential_network()

    test_split = 2000

    X_infer, y_infer = X_test[:test_split], y_test[:test_split]

    y_pred = net.infer(X_infer)

    assert y_pred.shape == y_infer.shape

    print("MSE Original: ", np.mean((y_pred - y_infer) ** 2))

    # KERAS COMP
    model = tf.keras.models.load_model(MODEL)

    y_pred_keras = model.predict(X_infer)

    assert y_pred_keras.shape == y_infer.shape

    print("MSE Keras: ", np.mean((y_pred_keras - y_infer) ** 2))

    # plot the 3 results
    print(y_pred.shape)
    fig = (
        px.line(
            x=np.arange(len(y_pred)),
            y=y_pred.reshape(-1),
        )
        .add_scatter(
            x=np.arange(len(y_pred_keras)),
            y=y_pred_keras.reshape(-1),
            mode="lines",
            name="Keras",
        )
        .add_scatter(
            x=np.arange(len(y_pred)),
            y=y_infer[:test_split].reshape(-1),
            mode="lines",
            name="Target",
        )
    )

    # save fig to html
    fig.write_html(str(BASE / "plot.html"))


if __name__ == "__main__":
    main()
