"""
Image Folder Monitor - Monitors a specified folder and automatically manages image files based on configurable parameters.

This module provides functionality to:
- Monitor a specified folder for image files
- Automatically delete oldest files when count exceeds maximum
- Read configuration from app config file
- Use watchdog for efficient file system monitoring
- Include proper logging and error handling
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.config_loader import load_config
from src.utils.logging_utils import get_logger


class ImageFolderMonitor:
    """Monitors a folder and manages image files based on configurable parameters."""
    
    def __init__(self, config_path: str = "config/app_config.yaml", folder_path: Optional[str] = None):
        """
        Initialize the image folder monitor.
        
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
        
        self.observer = Observer()
        self.event_handler = ImageFileHandler(self)
        
    def _get_folder_path_from_config(self) -> str:
        """Extract folder path from configuration."""
        try:
            # Try to get the photo capture path from app config
            return self.config.get('photo_capture', {}).get('image_save_path',
                                                          '/home/george/PycharmProjects/fruit-defect-detection/data/defect_320/test/images')
        except (AttributeError, KeyError):
            # Default path if configuration is not available
            return '/home/george/PycharmProjects/fruit-defect-detection/data/defect_320/test/images'
    
    def _get_max_images_from_config(self) -> int:
        """Extract maximum number of images from configuration."""
        try:
            return self.config.get('photo_capture', {}).get('max_images', 100)
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
        image_files = self.get_image_files()
        files_deleted = 0
        
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
        
        return files_deleted
    
    def start_monitoring(self):
        """Start monitoring the folder for changes."""
        self.logger.info(f"Starting to monitor folder: {self.folder_path}")
        self.logger.info(f"Max images allowed: {self.max_images}")
        self.logger.info(f"Monitoring formats: {', '.join(self.image_formats)}")
        
        # Schedule the event handler for the folder
        self.observer.schedule(self.event_handler, self.folder_path, recursive=False)
        self.observer.start()
        
        try:
            while True:
                # Check the folder at regular intervals
                time.sleep(30)  # Check every 30 seconds
                self.cleanup_old_images()
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def run_monitoring_loop(self):
        """Run the monitoring loop (separate method to allow integration with other systems)."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring the folder."""
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
        self.logger.info("Stopped monitoring folder")


class ImageFileHandler(FileSystemEventHandler):
    """Handles file system events for image files."""
    
    def __init__(self, monitor: ImageFolderMonitor):
        """
        Initialize the event handler.
        
        Args:
            monitor: The ImageFolderMonitor instance to use for cleanup
        """
        self.monitor = monitor
        self.logger = get_logger(__name__)
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            # Check if the created file is an image
            file_path = Path(event.src_path)
            ext = file_path.suffix.lower()[1:] if file_path.suffix else ''
            
            if ext in self.monitor.image_formats:
                self.logger.info(f"New image file detected: {event.src_path}")
                # Schedule cleanup after a short delay to avoid multiple rapid cleanups
                time.sleep(0.1)  # Small delay to ensure file is completely written
                self.monitor.cleanup_old_images()
    
    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory:
            # Check if the moved file is an image
            dest_path = Path(event.dest_path)
            ext = dest_path.suffix.lower()[1:] if dest_path.suffix else ''
            
            if ext in self.monitor.image_formats:
                self.logger.info(f"Image file moved to: {event.dest_path}")
                # Schedule cleanup after a short delay
                time.sleep(0.1)  # Small delay to ensure file is completely moved
                self.monitor.cleanup_old_images()


def main():
    """Main function to run the image folder monitor."""
    from src.utils.logging_utils import setup_logging
    
    # Set up logging
    setup_logging(level="INFO", log_file="logs/image_monitor.log")
    
    # Create and start the monitor
    try:
        monitor = ImageFolderMonitor()
        # Only perform initial cleanup when run as standalone, don't start continuous monitoring
        monitor.cleanup_old_images()
        print("Image folder monitor - initial cleanup completed. Use as part of main application for continuous monitoring.")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error running image folder monitor: {e}")
        raise


if __name__ == "__main__":
    main()