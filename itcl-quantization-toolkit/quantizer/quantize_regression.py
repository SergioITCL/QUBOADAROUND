import numpy as np
from itcl_quantizer.tensor_extractor.keras.keras_builder import build


if __name__ == "__main__":
    linespace = np.expand_dims(np.arange(-10, 10, 0.01), axis=1)

    build(
        "./models/keras/regression/reg_linear_out.h5", "abs_tanh_linear.json", linespace
    )
