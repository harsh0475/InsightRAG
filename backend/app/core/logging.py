"""Logging configuration for InsightRAG backend."""
import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """Configure system-wide logging with a standardized format.
    
    Args:
        log_level: Optional log level string (DEBUG, INFO, WARNING, ERROR).
                   Defaults to INFO if not provided.
    
    Returns:
        The configured root logger for the application.
    """
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_format = "%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Overwrite any pre-existing basicConfig
    )

    # Silence overly chatty external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger = logging.getLogger("insightrag")
    logger.setLevel(level)
    logger.info(f"Logging initialized at level: {level_name}")

    return logger

