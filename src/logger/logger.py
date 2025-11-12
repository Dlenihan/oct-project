import logging

from . import logger_config
from .logger_config import *  # noqa: F401,F403


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ['get_logger']
