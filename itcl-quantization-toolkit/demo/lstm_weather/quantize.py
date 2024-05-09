import tensorflow as tf
import numpy as np
from itcl_quantizer import keras_build
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent)
MODEL = f"{BASE_DIR}/models/2steps.h5"
DATA = f"{BASE_DIR}/data_2steps"
OUT_MODEL = f"{BASE_DIR}/qmodels/2steps.json"
x_train, y_train = np.load(f"{DATA}/x_train.npy"), np.load(
    f"{DATA}/y_train.npy"
)
x_test, y_test = np.load(f"{DATA}/x_test.npy"), np.load(
    f"{DATA}/y_test.npy"
)


rep_dataset_idx = int(len(x_train) * 0.3)

quant_model = keras_build(
    MODEL, OUT_MODEL, x_train[:rep_dataset_idx]
)


model = tf.keras.models.load_model(MODEL)
model.summary()

pred_y_test = model.predict(x_test)

np.save(f"{DATA}/pred_y_test.npy", pred_y_test)
