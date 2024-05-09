from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

BASE = str(Path(__file__).parent)
DATA = f"{BASE}/data_1_step"

ONNX_MODEL = f"{DATA}/model.onnx"
DYNAMIC_MODEL = f"{DATA}/dynamicq.onnx"
STATIC_MODEL = f"{DATA}/staticq.onnx"


class DataReader(CalibrationDataReader):
    def __init__(self):
        self.data = np.load(f"{DATA}/x_train.npy")
        self._idx = 0

    def get_next(self) -> dict | list | None:

        if self._idx >= len(self.data) * 0.5:
            return None

        idx = self._idx
        self._idx += 1
        return {
            "lstm_2_input:0": np.expand_dims(self.data[idx].astype(np.float32), axis=0)
        }


def convert_to_onnx():
    import tensorflow as tf

    from tf2onnx.convert import from_keras

    model = tf.keras.models.load_model(f"{DATA}/model.h5")
    onnx_model = from_keras(model, output_path=ONNX_MODEL)


def dynamic():
    quantize_dynamic(ONNX_MODEL, DYNAMIC_MODEL)


def static():
    print("static quant")
    quantize_static(
        ONNX_MODEL,
        STATIC_MODEL,
        DataReader(),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )


def test_model(model_path: str):

    x_test, y_test = (
        np.load(f"{DATA}/x_test.npy"),
        np.load(f"{DATA}/pred_y_test.npy"),
    )

    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    print("input_name", input_name)
    print("label_name", label_name)
    y_pred = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]

    assert y_pred.shape == y_test.shape, f"{y_pred.shape} != {y_test.shape}"

    # mse
    print(np.mean((y_pred - y_test) ** 2))


if __name__ == "__main__":
    static()
    test_model(STATIC_MODEL)
