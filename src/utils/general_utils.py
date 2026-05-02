import logging
import logging.config
import os
from os import PathLike

import yaml

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path: str | PathLike = "./config/logging.yaml",
    default_level: int = logging.INFO,
) -> None:
    try:
        os.makedirs("logs", exist_ok=True)
        with open(logging_config_path, encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())
        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is used.")


def to_native_path(path: str | PathLike) -> str:
    if path is None:
        raise TypeError("path cannot be None.")

    s = os.fspath(path=path).strip()
    if not s:
        return s

    if os.name == "nt":
        s = s.replace("/", "\\")
    else:
        s = s.replace("\\", "/")

    return os.path.normpath(path=s)
