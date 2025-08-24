import logging
import os
from datetime import datetime

# Directory to store log files
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Daily rotating filename (simple date-based naming)
LOG_FILE = os.path.join(
    LOGS_DIR,
    f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
)

# Configure root logger once
logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO
)

def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger inheriting the root configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger