import logging as _logging



import os
import sys

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()


_FORMAT = "[ %(levelname)s ] %(asctime)s.%(msecs)03d %(funcName)s %(message)s"
_DATE_FMT = "%H:%M:%S"
_logging.basicConfig(format=_FORMAT, level=LOGLEVEL, datefmt=_DATE_FMT, handlers=[])


def get_logger(name: str):
    logger_ = _logging.getLogger(name)
    logger_.handlers = []
    logger_.propagate = False
    formatter = _logging.Formatter(_FORMAT, _DATE_FMT)
    # Console logger config
    console_logger = _logging.StreamHandler(sys.stdout)
    console_logger.setLevel(LOGLEVEL)
    console_logger.setFormatter(formatter)

    # File logger config
    file_logger = _logging.FileHandler(f"log_{name}.txt".lower())
    file_logger.setLevel(level=_logging.DEBUG)
    file_logger.setFormatter(formatter)

    logger_.addHandler(console_logger)
    logger_.addHandler(file_logger)
    logger_.info("logger initialized")
    logger_.debug("logger initialized")
    logger_.warning("logger initialized")
    logger_.error("logger initialized")
    logger_.critical("logger initialized")
    return logger_
