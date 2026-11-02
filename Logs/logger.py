import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name=__name__, log_file="Logs/trainmodel.log"):
    logger = logging.getLogger(name=name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)s: %(lineno)d - %(message)s',datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)

    return logger



logger = setup_logger("Logs/trainmodel")