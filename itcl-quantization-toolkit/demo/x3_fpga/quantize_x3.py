import json
import logging
import os
import socket
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Thread
from time import perf_counter

import numpy as np
import typer
from itcl_inference_engine.network.sequential import Network as INetwork
from itcl_quantizer import keras_build
from itcl_quantizer.config.models.keras import QuantizerCfg

from demo_lib.fpga import InferFPGAMQTT, InputOutput, LossFPGA
from demo_lib.mqtt import TOPICS, create_client

BASE = Path(__file__).parent

BATCH = 5000


def infer_cpu_fn(net: INetwork, x: np.ndarray) -> np.ndarray:
    return net.infer(x)


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.square(x - y))  # type: ignore


class SmallModel:
    def __init__(self):
        self.model_path = BASE / "create_model/model_4l.h5"
        self.x_train = np.load(BASE / "create_model/x_train.npy")
        self.y_train = np.load(BASE / "create_model/y_train.npy")
        self.x = np.load(BASE / "create_model/x.npy")
        self.y = np.load(BASE / "create_model/y.npy")


class LargeModel:
    def __init__(self):
        self.model_path = BASE / "larger_model/model.h5"
        self.x_train = np.load(BASE / "larger_model/x_train.npy")
        self.y_train = np.load(BASE / "larger_model/y_train.npy")
        self.x = self.x_train
        self.y = self.y_train


class Accelerator(str, Enum):
    CPU = "cpu"
    FPGA = "fpga"
    FPGA_LOSS = "fpga-loss"


class ModelEnum(str, Enum):
    SMALL = "small"
    LARGE = "large"

    def resolve(self):
        return SmallModel() if self == ModelEnum.SMALL else LargeModel()


def main(
    accelerator: Accelerator = Accelerator.CPU,
    model: ModelEnum = ModelEnum.SMALL,
    batch: int = BATCH,
):

    selected_model = model.resolve()

    logging.getLogger("itcl_inference_engine").setLevel(logging.WARNING)
    logging.getLogger("itcl_quantizer").setLevel(logging.WARNING)

    client = create_client("quantizer", "admin", "admin", "localhost", 1883, 60)
    mqtt_thread = Thread(target=client.loop_forever)
    mqtt_thread.start()
    infer_fpga_client = InferFPGAMQTT(
        client,
        infer_topic=TOPICS.INFER,
        model_topic=TOPICS.MODEL_UPDATE,
        result_topic=TOPICS.RESULT,
        max_batch_size=batch,
    )

    def infer_fpga_fn(net: INetwork, x: np.ndarray) -> np.ndarray:
        res = infer_fpga_client(net, x)
        return res

    inference = infer_fpga_fn if accelerator == Accelerator.FPGA else infer_cpu_fn

    x_train, y_train = (
        selected_model.x_train[:batch],
        selected_model.y_train[:batch],
    )

    x, y = selected_model.x, selected_model.y
    out = BASE / "tmp.json"

    def loss_fn(net: INetwork) -> float:
        try:

            start = perf_counter()
            res = inference(net, x_train)

            mse_res = mse(res, y_train)
            print(f"Inferene Time: {perf_counter() - start}s")
            assert res.shape == y_train.shape
            return mse_res

        except Exception as e:
            logging.error(e)
            client.disconnect()
            os._exit(1)

    if accelerator == Accelerator.FPGA_LOSS:

        repr_dataset = InputOutput.from_numpy("test", x_train, y_train)
        client.publish(TOPICS.REPR_DATASET, repr_dataset.as_json(), qos=0)
        loss_fpga = LossFPGA(
            client,
            TOPICS.INFER_LOSS,
            TOPICS.MODEL_UPDATE,
        )

        def loss_fn(x):
            start = perf_counter()
            res = loss_fpga(x)
            print(f"Inferene Time: {perf_counter() - start}s")
            return res

    cfg = QuantizerCfg()

    # Disable equalizers
    # cfg.param_equalizer = None  # disable parameq
    cfg.ada_round_net = None  # disable adaround

    net = keras_build(
        str(selected_model.model_path), out.name, x_train, loss_fn=loss_fn, cfg=cfg
    )
    seq_net = net.as_sequential_network()

    print(f"Quant Loss: {loss_fn(seq_net)}")

    os._exit(0)

    predq = inference(seq_net, x)

    assert x.shape == predq.shape
    print(x.shape, y.shape)
    x_plot = np.squeeze(x)
    y_plot = np.squeeze(y)

    pred_q_plot = np.squeeze(predq)
    import plotly.express as px

    fig = px.line(
        x=x_plot, y=y_plot, title="Comparison of Quantized and Normal x^3 Functions"
    )
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        legend_title="Function Type",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.add_scatter(x=x_plot, y=pred_q_plot, name="Quantized")
    fig.write_html(str(BASE / "quantized.html"))

    test_input = x[:10000]
    test_res = predq[:10000]

    test_data = InputOutput.from_numpy("test", test_input, test_res)

    with open(BASE / "sample_data.json", "w") as f:
        f.write(test_data.as_json())

    client.disconnect()


if __name__ == "__main__":
    typer.run(main)
