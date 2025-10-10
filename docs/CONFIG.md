# Configuration Documentation

This document explains the purpose and structure of all configuration files used in the Fruit Defect Detection System.

## `config/model_config.yaml`

This file defines settings for the machine learning models used in the system.

### Structure

```yaml
# Model Configuration for Fruit Defect Detection System

# Fruit Detection Model Settings
fruit_detection:
  model_path: "models/fruit_detection/fruit_detector.pt"  # Path to the fruit detection model file
 confidence_threshold: 0.5  # Minimum confidence for fruit detection (0.0-1.0)
 target_classes:  # List of fruit classes the model should detect
    - apple
    - banana
    - tomato

# Defect Classification/Segmentation Model Settings
defect_detection:
  model_path: "models/defect_detection/defect_classifier.pt"  # Path to the defect detection model file
  confidence_threshold: 0.6  # Minimum confidence for defect classification (0.0-1.0)
  target_classes:  # List of defect classes the model should classify
    - defective
    - non_defective
```

## `config/app_config.yaml`

This file defines general application settings, including GUI, API, camera, photo capture, and debouncing configurations.

### Structure

```yaml
# Application Configuration for Fruit Defect Detection System

# Logging Settings
logging:
  level: "INFO" # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  file_path: "logs/app.log"  # Path to the log file

# GUI Settings
gui:
  show_gui: true  # Whether to show the GUI window (true/false)
  show_defect_status: true  # When true, shows both fruit type and defect status; when false, shows only fruit type

# Web API Settings
api:
  enabled: false  # Whether to enable API notifications (true/false)
 base_url: "http://localhost:8000"  # Base URL of the API
  endpoint: "/api/detections"  # API endpoint for sending detection data
  timeout: 10  # Request timeout in seconds

# Camera Settings
camera:
  source: 0  # Camera source (0 for default camera, or path to video file)
  width: 640  # Camera frame width in pixels
 height: 480  # Camera frame height in pixels
  fps: 30  # Camera frames per second

# Photo Capture Settings
photo_capture:
  capture_defective_only: false  # This setting no longer controls local capture - photos are always captured locally for logging; only affects API sending
  image_save_path: "captured_images/"  # Directory to save captured images
  image_format: "jpg" # Image format for saved photos (jpg, png, etc.)

# Note: All detected fruits (both defective and non-defective) are now captured locally for logging purposes,
# regardless of the capture_defective_only setting. This setting only controls whether images are sent to the API.

# Static Image Settings
static_image:
  enabled: false  # Set to true to use a static image instead of camera feed
  image_path: ""  # Path to the static image file to use when enabled

# Note: When static_image is enabled, the system will process the specified image once and then exit.
```

## `config/telegram_config.yaml`

This file defines settings for the Telegram bot integration.

### Structure

```yaml
# Telegram Bot Configuration for Fruit Defect Detection System

# Telegram Bot Settings
telegram:
  bot_token: "YOUR_TELEGRAM_BOT_TOKEN_HERE"  # Your Telegram bot token from BotFather
  enable_telegram: true  # Set to false to disable Telegram bot functionality
```

**Note:** You must replace `YOUR_TELEGRAM_BOT_TOKEN_HERE` with your actual Telegram bot token.

## `config/telegram_users.yaml`

This file defines a list of authorized user IDs who can receive notifications.

### Structure

```yaml
# Authorized Telegram User IDs for Fruit Defect Detection System

# List of authorized user IDs who can receive notifications
telegram_users:
  user_ids: 
    - 123456789  # Replace with actual Telegram user ID
    - 987654321  # Replace with actual Telegram user ID
```

**Note:** You must replace the example user IDs with actual Telegram user IDs of authorized users.