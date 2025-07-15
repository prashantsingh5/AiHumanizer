"""
Logging utilities for AI Humanizer
"""

import logging
import sys
from src.config import Config

def setup_logger():
    """Setup main application logger"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("ai_humanizer")

def get_logger(name: str):
    """Get logger for specific module"""
    return logging.getLogger(name)
