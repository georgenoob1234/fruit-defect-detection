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
## `config/trainer_config.yaml`

This file defines all training parameters for the fruit defect detection system using a comprehensive YAML-based configuration system.

### Structure

```yaml
# General Trainer Settings
trainer:
  enable_metrics: true
  enable_detailed_metrics: true
  enable_hardware_monitoring: true
  enable_ood_detection: true
  enable_stress_testing: true
  enable_calibration_metrics: true
  
  metrics_output_dir: "metrics"
  models_output_dir: "models"
  runs_output_dir: "runs"
  
  epochs: 100
  img_size: 320
  batch_size: 16
  learning_rate: 0.01
  save_period: 10

# Fruit Detection Model Configuration
fruit_detection:
  enabled: true
  model_path: "models/fruit_detection/fruit_detector.pt"
  model_name: "yolov8n.pt"
  dataset_path: "datasets/fruit_dataset.yaml"
  train_path: "datasets/fruit_dataset/images/train"
  val_path: "datasets/fruit_dataset/images/val"
  test_path: "datasets/fruit_dataset/images/test"
  
  epochs: 100
  img_size: 320
  batch_size: 16
  learning_rate: 0.01
  save_dir: "runs/detect/fruit_detector"
  
  num_classes: 3
  class_names: ["apple", "banana", "tomato"]
  confidence_threshold: 0.5

# Defect Classification Model Configuration
defect_classification:
  enabled: true
  model_path: "models/defect_classification/defect_classifier.pt"
  model_name: "yolov8n-cls.pt"
  dataset_path: "datasets/defect_classification_dataset.yaml"
  train_path: "datasets/defect_classification_dataset/images/train"
  val_path: "datasets/defect_classification_dataset/images/val"
  test_path: "datasets/defect_classification_dataset/images/test"
  
  epochs: 50
  img_size: 224
  batch_size: 32
  learning_rate: 0.001
  save_dir: "runs/classify/defect_classifier"
 
  num_classes: 2
  class_names: ["non_defective", "defective"]
  confidence_threshold: 0.6

# Defect Segmentation Model Configuration
defect_segmentation:
  enabled: true
  model_path: "models/defect_segmentation/defect_segmenter.pt"
  model_name: "yolov8n-seg.pt"
  dataset_path: "datasets/defect_segmentation_dataset.yaml"
  train_path: "datasets/defect_segmentation_dataset/images/train"
  val_path: "datasets/defect_segmentation_dataset/images/val"
  test_path: "datasets/defect_segmentation_dataset/images/test"
  
  epochs: 150
  img_size: 640
  batch_size: 8
  learning_rate: 0.001
  save_dir: "runs/segment/defect_segmenter"
  
  num_classes: 1
  class_names: ["defect"]
  confidence_threshold: 0.6

# Hardware and Performance Settings
hardware:
  use_gpu: true
  gpu_device: 0
  max_memory_usage_mb: 8192
  enable_memory_monitoring: true
  num_workers: 4
  pin_memory: true

# Metrics and Evaluation Settings
metrics:
  calculate_inference_time: true
  calculate_percentiles: [50, 99]
  calculate_precision_recall_f1: true
  calculate_per_class_metrics: true
  monitor_cpu_usage: true
  monitor_memory_usage: true
  monitor_gpu_usage: true
  calculate_model_size: true
  calculate_loading_time: true
  calculate_ood_metrics: true
  calculate_performance_distribution: true
  enable_stress_testing: true
  stress_test_conditions: ["glare", "shadows", "blur", "low_light"]
  calculate_calibration_metrics: true

# Logging and Output Settings
logging:
  level: "INFO"
  file_path: "logs/trainer.log"
  save_metrics_to_json: true
  save_metrics_to_csv: true
  timestamp_format: "%Y%m%d_%H%M%S"
```

**Note:** This configuration file centralizes all training parameters, eliminating the need for command-line arguments. All models can be trained with a single command: `python src/train_model.py`

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