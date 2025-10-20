# Scripts Documentation

This document provides an overview of all scripts in the `src/` directory of the Fruit Defect Detection System.

## Main Scripts

### `main.py`
The main entry point for the application. This script orchestrates all components of the fruit defect detection system, including camera handling, fruit detection, defect classification, API notifications, and Telegram bot integration. It implements a debouncing mechanism to prevent repeated notifications and handles graceful shutdown via signal handlers.

**Key Features:**
- Loads configuration from YAML files
- Initializes all system components
- Runs the main detection loop
- Implements debouncing for repeated detections
- Captures photos of ALL fruits (defective and non-defective) for logging
- Sends data to web API and Telegram bot
- Handles graceful shutdown
- Implements detection logging system for tracking detection history
- Updated debouncing mechanism that applies only to notifications, not photo capture

### `src/main_loop.py`
Contains the `MainLoop` class which encapsulates the core processing loop logic. This module coordinates between the camera, detectors, API handler, and Telegram bot, providing a clean separation of concerns for the main application loop.

**Key Components:**
- `MainLoop` class: Main application loop implementation
- Detection processing with debouncing
- Photo capture functionality (captures ALL fruits for logging, sends only defective to API)
- API and Telegram notification handling

## Detection Scripts

### `src/detection/fruit_detector.py`
Implements the fruit detection functionality using a YOLO model. Detects fruits in camera frames and returns bounding boxes and class information.

### `src/detection/defect_detector.py`
Implements the defect classification functionality. Takes fruit regions and determines if they are defective or not.

### `src/detection/detection_pipeline.py`
Orchestrates the detection process, coordinating between fruit detection and defect segmentation models.

## API Scripts

### `src/api/api_handler.py`
Handles communication with external web APIs, including sending detection data and image uploads. Supports both JSON-only and multipart form data requests.

## Camera Scripts

### `src/camera/camera_handler.py`
Manages camera operations including initialization, frame capture, and cleanup.

## Telegram Scripts

### `src/telegram/telegram_bot.py`
Handles Telegram bot operations for sending notifications with detection results and images. The module also includes a detection logging system that stores recent detection events and allows retrieval via the /showlogs command.

Key Features:
- Send text and image notifications to authorized users
- Support for multiple Telegram commands (/start, /help, /adduser, /showlogs)
- Detection logging system to store and retrieve recent detection history
- User authorization management

## Utility Scripts

### `src/utils/logging_utils.py`
Provides logging setup and configuration utilities.
### `src/utils/image_utils.py`
Contains image processing utilities used across the application.

### `src/utils/metrics_calculator.py`
Implements comprehensive metrics evaluation functionality for trained models. This module calculates and saves detailed model evaluation metrics including inference time, throughput, precision, recall, F1-score, accuracy, hardware usage, model size, OOD detection metrics, and performance distribution reports. Metrics are saved to timestamped JSON/CSV files in the /metrics folder.

### `src/train_model.py`
The main entry point for training models in the fruit defect detection system. This script now uses a comprehensive YAML-based configuration system instead of command-line arguments. The script handles training of the fruit detection model, defect classification model, and defect segmentation model with separate workflows for each model type. After each training session, the script automatically performs comprehensive metrics evaluation including inference time, throughput, precision, recall, F1-score, accuracy, hardware usage, model size, OOD detection metrics, and performance distribution reports. Metrics are saved to timestamped JSON/CSV files in the /metrics folder.

**Key Features:**
- YAML-based configuration system with centralized settings in `config/trainer_config.yaml`
- Independent training workflows for fruit detection (bounding boxes), defect classification (binary classification), and defect segmentation (segmentation masks)
- Automatic download of pre-trained models (yolov8n.pt, yolov8n-seg.pt, yolov8n-cls.pt) when needed
- Configurable training parameters (epochs, image size, batch size, learning rate) defined in the configuration file
- Automatic model saving in appropriate models subdirectories
- Comprehensive metrics evaluation after each training session
- Support for custom dataset formats with appropriate handling for detection (bounding boxes) and segmentation (masks)

### `src/config_loader.py`
Handles loading and parsing of YAML configuration files.

