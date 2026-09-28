"""Bounded operational logs. Transcript/evidence history lives in SQLite."""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(folder):
    logger = logging.getLogger('meeting_asr')
    if not logger.handlers:
        Path(folder).mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(Path(folder) / 'pipeline.log', maxBytes=2_000_000,
                                      backupCount=3, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if os.getenv('ASR_DEBUG_LLM') == '1' else logging.INFO)
        logger.propagate = False
    return logger
