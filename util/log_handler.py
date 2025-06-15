import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name="rag-n-rock", log_file=None):
    """
    Sets up a centralized logger used throughout the app.
    Prevents duplicate handlers.
    """
    LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
    os.makedirs(LOG_DIR, exist_ok=True)

    if not log_file:
        log_file = os.path.join(LOG_DIR, "app.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)

    # Avoid adding multiple handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # File handler (rotating logs)
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(name)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger