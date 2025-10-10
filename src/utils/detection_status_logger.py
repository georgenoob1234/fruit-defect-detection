"""
Detection Status Logger module for the Fruit Defect Detection System.

This module provides a logging system that tracks detection status with the following features:
- Logs detection start and end events
- Each log file is automatically named using the current timestamp in the format
  "log-[YYYY-MM-DD_HH-MM-SS].log"
- Preserves all previous log files without deletion, creating new files with unique
  timestamps for each session
"""

import logging
import time
from datetime import datetime
from pathlib import Path
import os


class DetectionStatusLogger:
    """
    A class to manage detection status logging.
    """
    
    def __init__(self, log_dir="logs"):
        """
        Initialize the Detection Status Logger.
        
        Args:
            log_dir (str): Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a new log file with current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = self.log_dir / f"detection_status_{timestamp}.log"
        
        # Initialize logging
        self.logger = self._setup_file_logger()
        
        # State tracking
        self.is_detecting = False
        
        # Log initial status
        self.logger.info(f"Detection Status Logger initialized. Log file: {self.log_file_path}")

    def _setup_file_logger(self):
        """
        Set up a file logger that writes only to the specified log file without console output.
        """
        logger_name = f"detection_status_{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file_path, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def detection_started(self):
        """
        Called when a detection starts (number of detections > 0).
        """
        was_detecting = self.is_detecting
        self.is_detecting = True
        
        if not was_detecting:
            # Detection just started, log status
            self.logger.info("detection started")

    def detection_ended(self):
        """
        Called when a detection ends (number of detections becomes 0).
        """
        was_detecting = self.is_detecting
        self.is_detecting = False
        
        if was_detecting:
            # Detection just ended, log status
            self.logger.info("detection ended")

    def get_current_log_file(self):
        """
        Get the path to the current log file.
        
        Returns:
            str: Path to the current log file
        """
        return str(self.log_file_path)

    def cleanup(self):
        """
        Clean up resources.
        """
        # Final log entry
        self.logger.info("Detection Status Logger shutdown")


def get_detection_status_logger():
    """
    Get a singleton instance of the DetectionStatusLogger.
    """
    if not hasattr(get_detection_status_logger, '_instance'):
        get_detection_status_logger._instance = DetectionStatusLogger()
    return get_detection_status_logger._instance