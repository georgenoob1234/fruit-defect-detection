"""
Background Image Monitor - Runs as a separate thread to periodically manage image files.

This module provides functionality to:
- Monitor a specified folder for image files at regular intervals
- Automatically delete oldest files when count exceeds maximum
- Read configuration from app config file
- Run as a background thread without interfering with main application
- Include proper logging and error handling
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from typing import List, Optional, Set

from src.config_loader import load_config
from src.utils.logging_utils import get_logger


class BackgroundImageMonitor:
    """Monitors a folder in the background and manages image files based on configurable parameters."""
    
    def __init__(self, config_path: str = "config/app_config.yaml", folder_path: Optional[str] = None):
        """
        Initialize the background image monitor.
        
        Args:
            config_path: Path to the configuration file
            folder_path: Optional folder path to monitor (overrides config if provided)
        """
        self.logger = get_logger(__name__)
        self.config = load_config(config_path)
        self.folder_path = folder_path or self._get_folder_path_from_config()
        self.max_images = self._get_max_images_from_config()
        self.image_formats = self._get_image_formats_from_config()
        
        # Validate folder path
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"Folder path does not exist: {self.folder_path}")
        
        # Thread control
        self.stop_event = Event()
        
    def _get_folder_path_from_config(self) -> str:
        """Extract folder path from configuration."""
        try:
            # Try to get the photo capture path from app config
            return self.config.get('photo_capture', {}).get('image_save_path',
                                                          '/home/george/PycharmProjects/fruit-defect-detection/captured_photos')
        except (AttributeError, KeyError):
            # Default path if configuration is not available
            return '/home/george/PycharmProjects/fruit-defect-detection/captured_photos'
    
    def _get_max_images_from_config(self) -> int:
        """Extract maximum number of images from configuration."""
        try:
            return self.config.get('photo_capture', {}).get('max_images', 10)
        except (AttributeError, KeyError):
            # Default to 100 if not specified
            return 100
    
    def _get_image_formats_from_config(self) -> Set[str]:
        """Extract image formats to monitor from configuration."""
        try:
            # Get the configured image format from photo capture settings
            config_format = self.config.get('photo_capture', {}).get('image_format', 'jpg')
            # Convert to lowercase and return as a set
            formats = {fmt.strip().lower() for fmt in config_format.split(',')}
            # Add common extensions for the same format if needed
            normalized_formats = set()
            for fmt in formats:
                if fmt in ['jpg', 'jpeg']:
                    # If the config specifies 'jpg', include both extensions
                    normalized_formats.add('jpg')
                    normalized_formats.add('jpeg')
                elif fmt in ['png', 'bmp', 'tiff', 'webp', 'gif']:
                    normalized_formats.add(fmt)
                else:
                    normalized_formats.add(fmt)
            return normalized_formats
        except (AttributeError, KeyError):
            # Default to common image formats if not specified
            return {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}
    
    def get_image_files(self) -> List[Path]:
        """Get all image files in the monitored folder, sorted by modification time (oldest first)."""
        image_files = []
        
        for file_path in Path(self.folder_path).iterdir():
            if file_path.is_file():
                # Get the file extension without the dot and convert to lowercase
                ext = file_path.suffix.lower()[1:] if file_path.suffix else ''
                if ext in self.image_formats:
                    image_files.append(file_path)
        
        # Sort by modification time (oldest first)
        image_files.sort(key=lambda x: x.stat().st_mtime)
        return image_files
    
    def cleanup_old_images(self) -> int:
        """
        Delete oldest image files if the count exceeds the maximum allowed.
        
        Returns:
            Number of files deleted
        """
        try:
            image_files = self.get_image_files()
            files_deleted = 0
            
            self.logger.debug(f"Found {len(image_files)} image files in {self.folder_path}")
            
            if len(image_files) > self.max_images:
                # Calculate how many files need to be deleted
                files_to_delete = len(image_files) - self.max_images
                
                self.logger.info(f"Found {len(image_files)} images, maximum allowed is {self.max_images}. "
                               f"Deleting {files_to_delete} oldest files.")
                
                # Delete the oldest files
                for i in range(files_to_delete):
                    file_to_delete = image_files[i]
                    try:
                        file_to_delete.unlink()  # Delete the file
                        self.logger.info(f"Deleted old image file: {file_to_delete}")
                        files_deleted += 1
                    except OSError as e:
                        self.logger.error(f"Failed to delete file {file_to_delete}: {e}")
            else:
                self.logger.debug(f"Image count ({len(image_files)}) is within limit ({self.max_images}), no cleanup needed")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup_old_images: {e}")
            raise
        
        return files_deleted

    def run_monitoring(self):
        """Run the monitoring loop in a separate thread."""
        self.logger.info(f"Starting background monitoring of folder: {self.folder_path}")
        self.logger.info(f"Max images allowed: {self.max_images}")
        self.logger.info(f"Monitoring formats: {', '.join(self.image_formats)}")
        
        # Perform initial cleanup
        try:
            self.cleanup_old_images()
        except Exception as e:
            self.logger.error(f"Error during initial cleanup: {e}")
        
        # Get cleanup interval from config, default to 30 seconds
        cleanup_interval = self.config.get('photo_capture', {}).get('cleanup_interval', 30)
        
        try:
            while not self.stop_event.is_set():
                # Wait for the configured interval or until stop event is set
                if self.stop_event.wait(timeout=cleanup_interval):
                    break # Stop event was set, exit the loop
                
                # Perform cleanup check
                try:
                    self.cleanup_old_images()
                except Exception as e:
                    self.logger.error(f"Error during periodic cleanup: {e}")
                    # Continue monitoring even if one cleanup cycle fails
                
        except Exception as e:
            self.logger.error(f"Error in background monitoring: {e}")
        finally:
            self.logger.info("Background image monitor stopped")

    def start_monitoring(self):
        """Start the monitoring in a background thread."""
        self.thread = Thread(target=self.run_monitoring, daemon=True)
        self.thread.start()
        return self.thread

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.stop_event.set()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
        self.logger.info("Background image monitoring stopped")


def main():
    """Main function to run the background image monitor."""
    from src.utils.logging_utils import setup_logging
    
    # Set up logging
    setup_logging(level="INFO", log_file="logs/background_image_monitor.log")
    
    # Create and start the monitor
    try:
        monitor = BackgroundImageMonitor()
        monitor.start_monitoring()
        
        # Keep the main thread alive to allow background monitoring
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping background image monitor...")
            monitor.stop_monitoring()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error starting background image monitor: {e}")
        raise


if __name__ == "__main__":
    main()