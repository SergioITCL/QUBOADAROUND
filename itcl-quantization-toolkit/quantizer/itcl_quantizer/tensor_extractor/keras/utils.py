import tensorflow as tf


class CheckType:
    @staticmethod
    def is_dense(layer):
        return isinstance(layer, tf.keras.layers.Dense)

    @staticmethod
    def is_input(layer):
        return isinstance(layer, tf.keras.layers.InputLayer) or isinstance(
            layer, tf.keras.layers.Input
        )

    @staticmethod
    def is_LSTM(layer):
        return isinstance(layer, tf.keras.layers.LSTM)

    @staticmethod
    def is_skippable(layer):
        for skippable in _SKIPPABLE:
            if isinstance(layer, skippable):
                return True
        return False


_SKIPPABLE = [
    tf.keras.layers.Dropout,
]


def get_layer_name(layer):
    return layer.name
