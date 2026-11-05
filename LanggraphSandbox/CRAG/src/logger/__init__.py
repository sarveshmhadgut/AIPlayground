"""
Configures application-wide logging with rotating file handlers.
"""

import logging
import os
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.exception import MyException

__all__: list[str] = [
    "BACKUP_COUNT",
    "LOGS_DIRNAME",
    "LOGS_DIRPATH",
    "LOG_FILENAME",
    "LOG_FILEPATH",
    "MAX_BYTES",
    "ROOT_DIRPATH",
    "config_logger",
    "get_current_timestamp",
    "logging",
]

# Root directory of CRAG
ROOT_DIRPATH: Path = Path(__file__).resolve().parents[2]
LOGS_DIRNAME: str = "logs"


def get_current_timestamp() -> str:
    """Generate formatted timestamp string for the current time.

    Returns:
        str: Timestamp formatted as DD-Mon-YY_HH-MM-SS.
    """
    return datetime.now(UTC).strftime("%d-%b-%y_%H-%M-%S")


LOG_FILENAME: str = f"{get_current_timestamp()}.log"
MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT: int = 4

try:
    LOGS_DIRPATH: Path = ROOT_DIRPATH / LOGS_DIRNAME
    os.makedirs(LOGS_DIRPATH, exist_ok=True)
    LOG_FILEPATH: Path = LOGS_DIRPATH / LOG_FILENAME
except Exception as e:
    raise MyException(e, sys) from e


def config_logger() -> None:
    """Configure the global logging settings with a rotating file handler.

    Raises:
        MyException: If configuring the logger fails.
    """
    try:
        logger: logging.Logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        file_format: logging.Formatter = logging.Formatter(
            "[ %(asctime)s ] %(name)s - %(filename)s - %(levelname)s - %(message)s"
        )

        if not logger.handlers:
            file_handler: RotatingFileHandler = RotatingFileHandler(
                str(LOG_FILEPATH),
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
    except Exception as e:
        raise MyException(e, sys) from e


config_logger()
