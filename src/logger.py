import logging
import os
from datetime import datetime

# Example: 01_28_2026_10_30_00.log File Name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Path for log file
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create log folder if not present
os.makedirs(logs_path, exist_ok=True)

# Full Log File Path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Logging Setup
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)