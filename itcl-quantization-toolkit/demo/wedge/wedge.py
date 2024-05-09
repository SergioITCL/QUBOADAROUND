# %%
from importlib import reload
from pathlib import Path

import itcl_quantizer
import numpy as np
import tensorflow as tf

reload(itcl_quantizer)
from itcl_quantizer import keras_build

base_path = str(Path(__file__).parent)


# %%
x_train, y_train = np.load(f"{base_path}/data/x_train.npy"), np.load(
    f"{base_path}/data/y_train.npy"
)

# shuffle the data
idx = np.arange(len(x_train))
np.random.shuffle(idx)
x_train, y_train = x_train[idx], y_train[idx]

model_path = f"{base_path}/models/low_complexity_model.h5"

# %%
model = tf.keras.models.load_model(model_path)

# %%
model.summary()


rep_dataset_idx = int(len(x_train) * 0.02)

# %%
keras_build(model_path, "tmp.json", x_train[:rep_dataset_idx])


# %%
