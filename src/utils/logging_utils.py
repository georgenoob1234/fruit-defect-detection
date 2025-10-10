"""
Logging utilities for the fruit detection system.
"""

import logging
import os
import threading
from datetime import datetime
from threading import Lock




def setup_logging(level="INFO", log_file="logs/app.log"):
    """
    Set up logging configuration for the application.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str): Path to the log file
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Set up formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Apply formatters to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure logging with standard handlers (no deduplication)
    # Clear any existing handlers to prevent duplicate logs
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set ultralytics logging level to WARNING to reduce verbosity
    try:
        import ultralytics
        ultralytics_logger = logging.getLogger('ultralytics')
        ultralytics_logger.setLevel(logging.WARNING)
    except ImportError:
        pass  # ultralytics not installed, skip this step


def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_detection(detection_data):
    """
    Log detection data in a structured format.
    
    Args:
        detection_data (dict): Detection information to log
    """
    logger = get_logger(__name__)
    logger.info(f"Detection: {detection_data}")


def log_error(error_msg, exception=None):
    """
    Log an error message, optionally with exception information.
    
    Args:
        error_msg (str): Error message to log
        exception (Exception, optional): Exception object to log
    """
    logger = get_logger(__name__)
    if exception:
        logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
    else:
        logger.error(error_msg)