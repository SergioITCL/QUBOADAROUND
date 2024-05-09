from abc import ABCMeta, abstractmethod


serializable_types = int | str | bytes

json = dict[
    str, serializable_types | list[serializable_types] | tuple[serializable_types]
]


class ISerializable(metaclass=ABCMeta):
    """Serializable Interface: Indicates that a class is serializable

    Args:
        metaclass (_type_, optional): _description_. Defaults to ABCMeta.
    """

    @abstractmethod
    def as_json(self) -> json:
        """Generates a DICT than can be serializable

        Returns:
            json: _description_
        """
