"""One logger for the whole pipeline, writing to a timestamped file under logs/.

Every component (ingestion, validation, transformation, trainer, pipeline,
app) imports `logging` after this module is imported once and just calls
logging.info(...)/logging.error(...) — a single consistent log format across
every stage, which matters once a training run fails somewhere in a 5-stage
pipeline and you need to find which stage and why.
"""
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
