from __future__ import annotations

import json
import logging
import threading
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from itcl_inference_engine.network.sequential import Network
from typing_extensions import Self

from .mqtt import Client, MQTTMessage, log_exception

E = TypeVar("E")

logger = logging.getLogger(__name__)


class InferFPGA(Protocol):
    @abstractmethod
    def __call__(self, net: Network, x: np.ndarray) -> Any:
        ...


class InferFPGAMQTT(InferFPGA):
    def __init__(
        self,
        mqtt_client: Client,
        result_topic: str,
        model_topic: str,
        infer_topic: str,
        max_batch_size: int,
        qos: int = 0,
        timeout: int = 3,
    ) -> None:

        self._client = mqtt_client
        self._client.subscribe(result_topic, qos=0)
        self._client.message_callback_add(result_topic, self.on_result)

        self._infer_topic = infer_topic
        self._model_topic = model_topic
        self._qos = qos

        # Async result synchronization
        self._result: np.ndarray | None = None
        self._result_event = threading.Event()
        self._max_batch_size = max_batch_size
        self._timeout = timeout

    @log_exception(logger=logger)
    def __call__(self, net: Network, x: np.ndarray) -> Any:

        json_net = json.dumps(net.schema)

        self._client.publish(self._model_topic, json_net, qos=self._qos)

        n_splits = len(x) // self._max_batch_size

        chunk_resuts = []

        for i, chunk in enumerate(np.array_split(x, n_splits)):
            # Clear the result sync event & buffer
            self._result_event.clear()
            self._result = None

            input_ = Input.from_numpy(str(i), chunk)
            input_json = json.dumps({"input": input_.as_kv()})
            self._client.publish(self._infer_topic, input_json, qos=self._qos)
            was_timeout = not self._result_event.wait(timeout=self._timeout)
            if self._result is None:
                raise RuntimeError(
                    f"Result is None, synchronization failed. {'Reason timeout' if was_timeout else ''}"
                )
            chunk_resuts.append(self._result)
        return np.concatenate(chunk_resuts)

    @log_exception(logger=logger)
    def on_result(self, _client: Client, _: None, message: MQTTMessage) -> None:
        res_dict = json.loads(message.payload.decode("utf-8"))
        res = Output.from_kv(res_dict)
        self._result = res.as_numpy()
        self._result_event.set()  # notify that a result is ready


class LossFPGA(InferFPGA):
    def __init__(
        self,
        mqtt_client: Client,
        result_topic: str,
        model_topic: str,
        qos: int = 0,
        timeout: int = 3,
    ):
        # attributes
        self._client = mqtt_client
        self._model_topic = model_topic
        self._qos = qos
        self._timeout = timeout

        self._result_event = threading.Event()

        self._loss: float = 0

        # subscribe to result topic
        self._client.subscribe(result_topic, qos=0)
        self._client.message_callback_add(result_topic, self.on_result)

    def __call__(self, net: Network) -> float:
        self._result_event.clear()
        json_net = json.dumps(net.schema)
        self._client.publish(self._model_topic, json_net, qos=self._qos)

        was_timeout = not self._result_event.wait(timeout=self._timeout)

        if was_timeout:
            raise RuntimeError("Timeout, no result received")

        return self._loss

    def on_result(self, _client: Client, _: None, message: MQTTMessage) -> None:
        res_dict = json.loads(message.payload.decode("utf-8"))
        self._loss = float(res_dict["loss"])
        self._result_event.set()


@dataclass
class Input(Generic[E]):

    id: str
    shape: list[int]
    tensor: list[E]

    @classmethod
    def from_numpy(cls, id: str, batch: np.ndarray) -> Self:

        flattened: list[E] = batch.flatten().tolist()

        return cls(
            id=id, shape=[len(batch), len(flattened) // len(batch)], tensor=flattened
        )

    def as_kv(self) -> dict[str, Any]:
        return {"id": self.id, "shape": self.shape, "tensor": self.tensor}

    def as_json(self) -> str:
        return json.dumps(self.as_kv())


@dataclass
class Output(Generic[E]):
    id: str
    shape: list[int]
    tensor: list[E]

    @classmethod
    def from_kv(cls, kv_dict: dict[str, Any]) -> Self:
        return cls(kv_dict["id"], kv_dict["shape"], kv_dict["tensor"])

    def as_kv(self) -> dict[str, Any]:
        return {"id": self.id, "shape": self.shape, "tensor": self.tensor}

    @classmethod
    def from_numpy(cls, id: str, batch: np.ndarray) -> Self:
        flattened: list[E] = batch.flatten().tolist()

        return cls(
            id=id, shape=[len(batch), len(flattened) // len(batch)], tensor=flattened
        )

    def as_numpy(self) -> np.ndarray:
        return np.array(self.tensor).reshape(self.shape)


@dataclass
class InputOutput:
    input: Input
    output: Output

    @classmethod
    def from_numpy(cls, id: str, x: np.ndarray, y: np.ndarray) -> Self:
        input = Input.from_numpy(id, x)
        output = Output.from_numpy(id, y)

        return cls(input, output)

    def as_kv(self) -> dict[str, Any]:
        return {"input": self.input.as_kv(), "output": self.output.as_kv()}

    def as_json(self) -> str:
        return json.dumps(self.as_kv())
