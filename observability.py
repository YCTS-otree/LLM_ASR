"""Bounded operational logs. Transcript/evidence history lives in SQLite."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(folder):
    logger = logging.getLogger('meeting_asr')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        Path(folder).mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(Path(folder) / 'pipeline.log', maxBytes=2_000_000,
                                      backupCount=3, encoding='utf-8')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if sys.stderr is not None and not any(getattr(h, '_asr_console', False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stderr)
        console._asr_console = True
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.setLevel(logging.DEBUG if os.getenv('ASR_DEBUG_LLM') == '1' else logging.INFO)
    logger.propagate = False
    return logger
