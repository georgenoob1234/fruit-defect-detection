# Fruit Defect Detection System

Real-time fruit defect detection system using camera input or static images to identify and segment external defects on apples, bananas, and tomatoes. The system combines advanced computer vision with YOLO models to provide accurate detection and classification of fruit defects.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Telegram Bot Integration](#telegram-bot-integration)
- [GUI Interface](#gui-interface)
- [Training Models](#training-models)
- [Troubleshooting](#troubleshooting)
- [Documentation Links](#documentation-links)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Fruit Defect Detection System is a comprehensive solution that combines computer vision, machine learning, and real-time processing to detect and classify fruit defects. The system uses a multi-model approach with YOLO models for both fruit detection and defect segmentation, providing accurate and reliable results.

### Key Components
- **Fruit Detection Model**: Detects fruit types (apple, banana, tomato) with bounding boxes
- **Defect Segmentation Model**: Creates precise segmentation masks for defects on detected fruits
- **Camera Interface**: Real-time processing from camera feed or static images
- **GUI Interface**: Visual display of detection results with bounding boxes and defect overlays
- **API Integration**: Send detection data to external web services
- **Telegram Bot**: Real-time notifications with images of defective fruits

## Features

- Real-time fruit detection and defect classification
- Support for apples, bananas, and tomatoes
- Visual GUI with bounding boxes and defect segmentation overlays
- Photo capture of all detected fruits for logging
- Configurable confidence thresholds
- Telegram bot integration for notifications
- Web API integration for remote data collection
- Static image processing mode
- Detection logging system
- Multi-model architecture for accuracy
- Comprehensive metrics evaluation including inference time, throughput, precision, recall, F1-score, accuracy, hardware usage, model size, OOD detection metrics, and performance distribution reports
- Metrics saved in timestamped JSON/CSV format in /metrics folder
- KPI validation and stress testing for challenging conditions

## System Requirements

### Hardware Requirements
- Modern CPU (Intel i5 or equivalent recommended)
- GPU support for faster inference (NVIDIA GPU with CUDA support recommended)
- Minimum 4GB RAM (8GB+ recommended)
- USB camera or video file input
- 2GB free disk space for models and logs

### Software Requirements
- Python 3.8 or higher
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLOv8
- YAML parser
- NumPy
- Telegram Bot API support

### Supported Operating Systems
- Linux (Ubuntu 18.04+, Debian 10+)
- Windows 10/11
- macOS (10.15+)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fruit-defect-detection.git
cd fruit-defect-detection
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models
The system requires two pre-trained models:
- Fruit detection model: `models/fruit_detection/fruit_detector.pt`
- Defect segmentation model: `models/defect_detection/defect_segmenter.pt`

These models can be trained using the provided training script or obtained through other means.

### 5. Set Up Configuration Files
Update the configuration files in the `config/` directory with your specific settings (see [Configuration](#configuration) section).

## Configuration

The system uses multiple YAML configuration files located in the `config/` directory:

### `config/model_config.yaml`
Configures the machine learning models:
```yaml
# Fruit Detection Model Settings
fruit_detection:
  model_path: "models/fruit_detection/fruit_detector.pt"
  confidence_threshold: 0.5
  target_classes:
    - apple
    - banana
    - tomato

# Defect Segmentation Model Settings
defect_detection:
  model_path: "models/defect_detection/defect_segmenter.pt"
  confidence_threshold: 0.6
  target_classes:
    - defective
    - non_defective
```

### `config/app_config.yaml`
Configures general application settings:
```yaml
# Logging Settings
logging:
 level: "INFO"
  file_path: "logs/app.log"

# GUI Settings
gui:
  show_gui: true
  show_defect_status: true

# Web API Settings
api:
  enabled: false
  base_url: "http://localhost:8000"
  endpoint: "/api/detections"
  timeout: 10

# Camera Settings
camera:
  source: 0
  width: 640
  height: 480
  fps: 30

# Photo Capture Settings
photo_capture:
  capture_defective_only: false
  image_save_path: "captured_images/"
  image_format: "jpg"

# Static Image Settings
static_image:
  enabled: false
  image_path: ""
```

### `config/telegram_config.yaml`
Configures Telegram bot integration:
```yaml
telegram:
  bot_token: "YOUR_TELEGRAM_BOT_TOKEN_HERE"
  enable_telegram: true
```

### `config/telegram_users.yaml`
Lists authorized Telegram user IDs:
```yaml
telegram_users:
  user_ids: 
    - 123456789
    - 987654321
```

## Usage

### Running the Application
```bash
python main.py
```

### Static Image Mode
To process a static image instead of camera feed:
1. Update `config/app_config.yaml`:
   ```yaml
   static_image:
     enabled: true
     image_path: "/path/to/your/image.jpg"
   ```
2. Run the application:
   ```bash
   python main.py
   ```

### Keyboard Controls
- Press 'Q' to quit the application
- Close the GUI window to exit

### Telegram Bot Commands
- `/start` - Start interacting with the bot
- `/help` - Show available commands
- `/adduser <username|user_id>` - Add a new user (admin only)
- `/showlogs [count]` - Show recent detection logs (admin only)

## API Integration

The system can send detection data to external web APIs. To enable API integration:

1. Update `config/app_config.yaml`:
   ```yaml
   api:
     enabled: true
     base_url: "http://your-api-server.com"
     endpoint: "/api/detections"
     timeout: 10
   ```

2. The system sends JSON data in the following format:
   ```json
   {
     "fruit_class": "apple",
     "is_defective": true,
     "confidence": 0.85,
     "timestamp": "2023-12-01T10:30:45.123456",
     "bbox": [x1, y1, x2, y2]
   }
   ```

## Telegram Bot Integration

The system can send real-time notifications via Telegram. To set up:

1. Create a Telegram bot with [@BotFather](https://t.me/BotFather)
2. Update `config/telegram_config.yaml` with your bot token
3. Add authorized user IDs to `config/telegram_users.yaml`
4. Ensure users have started a chat with your bot by sending any message to it

When a defective fruit is detected, the system will send a notification with:
- Fruit type and defect status
- Confidence score
- Timestamp
- Image of the defective fruit

## GUI Interface

The system features a graphical user interface that displays:
- Live camera feed
- Bounding boxes around detected fruits (green for normal, red for defective)
- Defect segmentation overlays when defects are detected
- Confidence scores for each detection
- Overall detection statistics
- Application controls

The GUI can be enabled/disabled through the configuration file.

## Training Models

To train custom models for your specific use case, use the training script with the new YAML-based configuration system:

### Basic Training
```bash
python src/train_model.py
```

The script will automatically load the configuration from `config/trainer_config.yaml` and train all enabled models according to the specified parameters.

### Training Specific Model Types
```bash
python src/train_model.py --model_type fruit_detection
```

Available model types:
- `fruit_detection`: Trains the fruit detection model
- `defect_classification`: Trains the defect classification model
- `defect_segmentation`: Trains the defect segmentation model

### Configuration
All training parameters are now defined in `config/trainer_config.yaml`. This includes:
- Model-specific configurations for fruit detection, defect classification, and defect segmentation
- Automatic derivation of train, validation, and test paths from dataset YAML file location
- Hardware and performance settings
- Metrics and evaluation settings with master metrics toggle
- Logging and output settings

### Metrics Evaluation
After training, the system automatically performs comprehensive metrics evaluation including:
- Inference time metrics (p50/p90/p99 percentiles)
- Throughput in frames per second (FPS)
- Precision, recall, F1-score, and accuracy for overall and per-class metrics
- Hardware metrics (GPU/CPU usage and memory consumption)
- Model size and loading time
- OOD (Out-of-Distribution) detection metrics
- Performance distribution reports
- Calibration metrics and stress testing
- KPI validation against targets (recall ≥0.80, precision ≥0.85)

Metrics are saved to timestamped JSON and CSV files in the `/metrics` folder. The JSON format preserves the hierarchical structure of metrics, while the CSV format provides a flattened view with dot notation (e.g., "classification_metrics.overall.precision").

For detailed training instructions, see the [Training Documentation](docs/TRAINER.md).

## Troubleshooting

### Common Issues

#### Camera Not Working
- Verify camera is properly connected and not being used by another application
- Check camera permissions on your system
- Try changing the camera source in `config/app_config.yaml` (try values like 0, 1, or '/dev/video0')

#### Model Loading Errors
- Ensure model files exist at the specified paths in `config/model_config.yaml`
- Verify model file integrity
- Check that PyTorch and Ultralytics are properly installed

#### Telegram Notifications Not Sending
- Verify the bot token is correct in `config/telegram_config.yaml`
- Ensure user IDs in `config/telegram_users.yaml` are correct
- Check that users have started a chat with the bot (send any message to it first)
- Verify the `enable_telegram` option is set to `true`

#### Low Detection Accuracy
- Try adjusting confidence thresholds in `config/model_config.yaml`
- Consider retraining models with more diverse dataset
- Ensure proper lighting conditions during detection

#### High CPU Usage
- Reduce camera resolution in `config/app_config.yaml`
- Lower camera FPS in `config/app_config.yaml`
- Consider using GPU acceleration if available

### Performance Optimization
- Use GPU acceleration when available
- Optimize camera resolution and FPS settings
- Adjust model input size in model configuration
- Consider using lighter model variants for real-time applications

## Documentation Links

This project includes comprehensive documentation files to help you understand and use all features:

- [**Configuration Documentation**](docs/CONFIG.md) - Details about all configuration files and their settings, including model, application, and Telegram configurations.

- [**Telegram Bot Integration Documentation**](docs/TELEGRAM_BOT.md) - Complete guide to setting up and using the Telegram bot integration, including commands, setup process, and troubleshooting.

- [**GUI Handler Documentation**](docs/GUI_HANDLER.md) - Documentation for the graphical user interface, including features, components, and configuration options.

- [**Web API Integration Documentation**](docs/WEB_API.md) - Instructions for integrating with external web APIs, including data format, configuration, and error handling.

- [**Detection Logging System Documentation**](docs/DETECTION_LOGGING.md) - Information about the detection logging system and how to retrieve logs via Telegram commands.

- [**Scripts Documentation**](docs/SCRIPTS.md) - Overview of all scripts in the `src/` directory with their purposes and key features.

- [**Training Script Documentation**](docs/TRAINER.md) - Comprehensive guide to training custom models, including installation requirements, usage instructions, and troubleshooting.

## Contributing

We welcome contributions to improve the Fruit Defect Detection System! To contribute:

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Create a new branch for your feature or bug fix

### Contribution Guidelines
- Follow Python PEP 8 coding standards
- Write clear, descriptive commit messages
- Add documentation for new features
- Ensure all tests pass before submitting
- Create detailed pull requests with explanations

### Reporting Issues
- Use the GitHub issue tracker
- Provide detailed information about the issue
- Include steps to reproduce the problem
- Specify your environment (OS, Python version, etc.)

### Feature Requests
- Open an issue with your feature request
- Explain the use case and benefits
- Consider implementation complexity
- Be open to discussion and feedback

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details.

The GPL-3.0 is a copyleft license that ensures the software and all its derivatives remain free and open source. Anyone who uses, modifies, or distributes this software must make their code also free and open source under the same terms, and cannot relicense it under a proprietary license. This license also prohibits selling or making profit from the software without making the source code freely available.

## Acknowledgments

- YOLOv8 models from Ultralytics for the core detection functionality
- OpenCV for computer vision operations
- PyTorch for deep learning capabilities
- Telegram Bot API for notification services
- All contributors and users of this system
