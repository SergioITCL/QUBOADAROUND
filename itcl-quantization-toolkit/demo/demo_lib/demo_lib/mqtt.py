from logging import Logger
from paho.mqtt.client import Client, MQTTMessage


class TOPICS:
    MODEL_UPDATE = "INFERENCE_ENGINE/MODEL/UPDATE"
    INFER = "PREPROCESSING/OUT/JSON"
    RESULT = "INFERENCE_ENGINE/OUT/JSON"
    REPR_DATASET = "INFERENCE_ENGINE/DATASET/SET"
    INFER_LOSS = "INFERENCE_ENGINE/OUT/LOSS"


def create_client(
    name: str,
    user: str,
    pwd: str,
    host: str,
    port: int,
    keep_alive: int,
) -> Client:

    print(f"Connecting to MQTT broker {host}:{port} as {name}")
    client = Client(client_id=name, clean_session=True)
    client.reinitialise(client_id=name, clean_session=True)
    client.username_pw_set(user, pwd)
    client.connect(host, port, keep_alive)
    client.suppress_exceptions = True  # Do not die on an exception inside a callback
    print(f"Connected to MQTT Broker: {host}:{port} with client_id: {name}")
    return client


def log_exception(logger: Logger):
    """Decorator that logs a function's exception.

    Args:
        logger (Logger): A python logger
    """

    def inner_decorator(f):
        if logger is None or f is None:
            raise ValueError("logger and function cannot be None")

        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                logger.exception(e, stack_info=True, exc_info=True)
                return None

        return wrapped

    return inner_decorator
