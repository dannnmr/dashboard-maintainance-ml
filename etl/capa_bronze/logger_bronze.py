# logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

# Logs folder relative to this file (works locally and inside containers)
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "bronze.log")

logger = logging.getLogger("bronze_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
