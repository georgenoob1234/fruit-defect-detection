# Unified Photo Capture System

The Unified Photo Capture System combines the original photo capturer with an automatic image folder monitor that runs in the background. This system ensures that photos continue to be captured as before, while automatically managing storage space by maintaining only the specified maximum number of images.

## Features

- **Original Photo Capturer**: Continues to function as before, capturing new images based on detection results
- **Background Image Monitor**: Runs as a separate thread to periodically check and manage image files
- **Automatic Image Management**: Monitors a folder for image files and automatically deletes the oldest files when the count exceeds the maximum allowed
- **Configurable Parameters**: Maximum number of images and image formats to monitor can be configured
- **Non-Interfering Operation**: Both systems work in parallel without affecting each other
- **Chronological Deletion**: Maintains chronological order, removing the oldest files first while preserving the most recent ones
- **Configurable Settings**: Reads configuration from the app configuration file
- **Real-time Monitoring**: Uses the watchdog library to monitor file system events in real-time (for the original ImageFolderMonitor component)

## Configuration

The unified system uses the following settings from `config/app_config.yaml`:

```yaml
# Photo Capture Settings
photo_capture:
  capture_defective_only: true  # When true, only captures photos of defective fruits; when false, captures all detected fruits
  image_save_path: "/home/george/PycharmProjects/fruit-defect-detection/captured_photos"  # Directory where captured photos will be saved
  image_format: "jpg" # Format for saved images (jpg, png, etc.) - can be comma-separated list
  max_images: 10  # Maximum number of images to retain in the folder
```

## How It Works

The unified system operates in two parallel components:

1. **Original Photo Capturer**: Functions exactly as before, detecting fruits and defects, then capturing images when conditions are met
2. **Background Image Monitor**: Runs in a separate thread, periodically checking the image folder and automatically deleting the oldest photos when the total number reaches the configured limit (100 by default)

Both systems work in unison without interfering with each other - the photo capturer continues capturing new images while the monitor handles cleanup, ensuring continuous operation and storage management.

## Usage

### As a Standalone Program

To run the image folder monitor as a standalone program:

```bash
python -m src.utils.image_folder_monitor
```

This will perform an initial cleanup based on the configuration file.

### As a Module

To use the ImageFolderMonitor class in your code:

```python
from src.utils.image_folder_monitor import ImageFolderMonitor

# Initialize with default configuration
monitor = ImageFolderMonitor()

# Or specify a custom config path and folder
monitor = ImageFolderMonitor(
    config_path="config/app_config.yaml",
    folder_path="/path/to/your/images"
)

# Manually trigger a cleanup
deleted_count = monitor.cleanup_old_images()
print(f"Deleted {deleted_count} old images")

# Start real-time monitoring
monitor.start_monitoring()
```

### Manual Cleanup

You can also use the monitor for one-time cleanup operations:

```python
from src.utils.image_folder_monitor import ImageFolderMonitor

monitor = ImageFolderMonitor()
deleted_count = monitor.cleanup_old_images()
print(f"Deleted {deleted_count} old images")
```

## Supported Image Formats

By default, the monitor handles these image formats:
- JPG/JPEG
- PNG
- BMP
- TIFF
- WEBP

The specific formats can be configured in the app configuration file.

## Event Handling

The monitor responds to the following file system events:

- **File Creation**: When a new image file is created in the monitored folder, the monitor checks if cleanup is needed
- **File Movement**: When an image file is moved into the monitored folder, the monitor checks if cleanup is needed

## Logging

The unified system uses the same logging system as the rest of the application. Logs are written to:

- Console output
- Log file (as configured in the app configuration)

Log levels can be adjusted in the main configuration file.

## Integration

The unified photo capture system is integrated into the main application (`main.py`) where both the original photo capturer and the background image monitor run simultaneously. The background monitor runs as a separate thread that periodically checks the image folder and maintains the configured maximum number of images without interfering with the ongoing photo capture process. The main application initializes the monitor and calls cleanup operations after each photo capture to ensure storage space is managed automatically without manual intervention. This helps prevent storage issues when capturing many images over time.