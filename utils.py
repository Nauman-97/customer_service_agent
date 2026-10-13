from pathlib import Path
import logging

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str, filename: str, level:str):

    levels = {"debug":logging.DEBUG,
              "info":logging.INFO,
              "warning":logging.WARNING,
              "error": logging.ERROR,
              "critical" :logging.CRITICAL}

    logger = logging.getLogger(name)

    # avoid duplicate handlers
    if not logger.handlers:

        handler = logging.FileHandler(
            LOG_DIR / filename
        )

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

        logger.setLevel(levels[level])

        logger.propagate = False

    return logger