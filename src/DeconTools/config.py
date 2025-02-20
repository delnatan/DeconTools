import logging

logger = logging.getLogger("DeconTools")


def set_logging_level(level: str | int) -> None:
    """set logging level

    Args:
        level: Can be either a string ('DEBUG', 'INFO', 'WARNING', 'ERROR',
        'CRITICAL') or an integer (10, 20, 30, 40, 50) corresponding to the
        logging levels.
    """

    if isinstance(level, str):
        level = level.upper()
        numeric_level = getattr(logging, level, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
    elif isinstance(level, int):
        numeric_level = level
    else:
        raise TypeError("Level must be a string or an integer")

    logger.setLevel(numeric_level)

    # Create a StreamHandler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"Logging level set to {logging.getLevelName(numeric_level)}")


set_logging_level("INFO")
