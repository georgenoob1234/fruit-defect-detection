# Training Script Documentation

## Overview

This documentation covers the training script `src/train_model.py`, which is designed for training the multi-model setup of the fruit defect detection system. The script now uses a comprehensive YAML-based configuration system instead of command-line arguments. The script handles training of the fruit detection model, defect classification model, and defect segmentation model with separate workflows for each model type. After each training session, the script automatically performs comprehensive metrics evaluation.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation Requirements](#installation-requirements)
3. [Usage Instructions](#usage-instructions)
4. [YAML Configuration System](#yaml-configuration-system)
5. [Configuration Options](#configuration-options)
6. [Multi-Model Training Process](#multi-model-training-process)
7. [Examples](#examples)
8. [Customization Guide](#customization-guide)
9. [Model Management](#model-management)
10. [Troubleshooting](#troubleshooting)

## Introduction

The fruit defect detection system employs a multi-model approach:

1. **Fruit Detection Model**: A YOLOv8n detection model that detects fruit types (apple, banana, tomato) with bounding boxes
2. **Defect Segmentation Model**: A YOLOv8n-seg model that creates precise segmentation masks for defects on detected fruits

The training script allows independent training of each model with configurable parameters and appropriate dataset format handling for both detection (bounding boxes) and segmentation (masks).

## Installation Requirements

Before running the training script, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

The key dependencies required for training are:
- ultralytics
- opencv-python
- PyYAML
- numpy
- torch

## Usage Instructions

The training script now uses a comprehensive YAML-based configuration system. All training parameters are defined in `config/trainer_config.yaml`.

### Basic Usage

To train all enabled models according to the configuration:
```bash
python src/train_model.py
```

To train a specific model type:
```bash
python src/train_model.py --model_type fruit_detection
```

Available model types:
- `fruit_detection`: Trains the fruit detection model
- `defect_classification`: Trains the defect classification model
- `defect_segmentation`: Trains the defect segmentation model

## YAML Configuration System

The training script now uses a comprehensive YAML-based configuration system instead of command-line arguments. All training parameters are defined in `config/trainer_config.yaml`.

### Configuration File Structure

The configuration file is organized into several sections:

1. **General Trainer Settings**: Global settings for the trainer
2. **Model-Specific Configurations**: Separate sections for fruit detection, defect classification, and defect segmentation
3. **Hardware and Performance Settings**: GPU/CPU settings and performance optimizations
4. **Metrics and Evaluation Settings**: Configuration for metrics calculation and evaluation
5. **Logging and Output Settings**: Logging levels and output formats

### Example Configuration

```yaml
# General Trainer Settings
trainer:
  # Master toggle switch for enabling/disabling all metric functions
  # When set to false, no metrics will be calculated regardless of other metric-related configuration values
  enable_metrics: true
  enable_detailed_metrics: true  # Enables comprehensive metrics including inference time, throughput, etc.
  enable_hardware_monitoring: true
  enable_ood_detection: true
  enable_stress_testing: true
  enable_calibration_metrics: true
  
  metrics_output_dir: "metrics"
  models_output_dir: "models"
  runs_output_dir: "runs"
  
  # Validation settings
  validation_split: 0.2  # 20% of training data used for validation
  test_split: 0.1        # 10% of data used for testing
  
  # Performance targets
  target_inference_time_ms: 200  # Target ≤20ms per frame
  target_end_to_end_latency: 0.5 # Target ≤0.5s end-to-end latency
  
  # KPI targets
  min_recall_problematic_cases: 0.80
  min_precision_marked_items: 0.85
  max_unnecessary_suggestions: 0.10
  
  # Minimum requirements for test set
  min_test_images: 120
  min_images_per_class: 40
  min_problematic_examples: 30
# Fruit Detection Model Configuration
fruit_detection:
  enabled: true
  model_path: "models/fruit_detection/fruit_detector.pt"
  model_name: "yolov8n.pt"
  dataset_path: "datasets/fruit_dataset.yaml"
  # Note: train_path, val_path, and test_path are automatically derived from dataset_path
  # but can be overridden if needed
  
  epochs: 100
  img_size: 320
  batch_size: 16
  learning_rate: 0.01
  save_period: 10  # Save checkpoint every N epochs
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
  # Note: train_path, val_path, and test_path are automatically derived from dataset_path
  # but can be overridden if needed
  
  epochs: 50
  img_size: 224
  batch_size: 32
  learning_rate: 0.001
  save_period: 10  # Save checkpoint every N epochs
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
  # Note: train_path, val_path, and test_path are automatically derived from dataset_path
  # but can be overridden if needed
  
  epochs: 150
  img_size: 640
  batch_size: 8
  learning_rate: 0.001
  save_period: 10  # Save checkpoint every N epochs
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
  # Note: All metrics calculation is controlled by the master enable_metrics toggle in the trainer section
  # When trainer.enable_metrics is false, no metrics will be calculated regardless of these settings
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

### Usage Instructions

To train models using the YAML configuration:

```bash
python src/train_model.py
```

The script will automatically load the configuration from `config/trainer_config.yaml` and train all enabled models according to the specified parameters.

To train a single model type:

```bash
python src/train_model.py --model_type fruit_detection
```

Available model types:
- `fruit_detection`: Trains the fruit detection model
- `defect_classification`: Trains the defect classification model
- `defect_segmentation`: Trains the defect segmentation model

## Configuration Options

All configuration options are now defined in the YAML configuration file `config/trainer_config.yaml`. See the [YAML Configuration System](#yaml-configuration-system) section for detailed information about the structure and parameters.

## Multi-Model Training Process

### Fruit Detection Model Training

The primary model training process involves:

1. Loading a pre-trained YOLOv8n-seg model (automatically downloaded if not present)
2. Configuring the model for fruit detection (3 classes: apple, banana, tomato)
3. Training the model on your dataset to detect fruits and propose defect regions
4. Saving the trained model in the appropriate models subdirectory

### Defect Segmentation Model Training

The defect segmentation model training process involves:

1. Loading a pre-trained YOLOv8n-seg model (automatically downloaded if not present)
2. Configuring the model for defect segmentation with pixel-level mask generation
3. Training the model on your dataset to create precise segmentation masks for defects
4. Saving the trained model in the appropriate models subdirectory


## Examples

### Example 1: Modifying the Trainer Configuration

To customize training parameters, modify the `config/trainer_config.yaml` file:
```yaml
# Example: Increase epochs and adjust learning rate for fruit detection
fruit_detection:
  enabled: true
  model_path: "models/fruit_detection/fruit_detector.pt"
  model_name: "yolov8n.pt"
  dataset_path: "datasets/fruit_dataset.yaml"
  # Note: train_path, val_path, and test_path are automatically derived from dataset_path
  # but can be overridden if needed
  
  epochs: 200  # Increased from default 100
  img_size: 384
  batch_size: 8
  learning_rate: 0.005  # Decreased from default 0.01
  save_period: 10  # Save checkpoint every N epochs
 save_dir: "runs/detect/fruit_detector"
  
  num_classes: 3
  class_names: ["apple", "banana", "tomato"]
  confidence_threshold: 0.5

```

### Example 2: Enabling/Disabling Specific Models

To train only specific models, modify the `enabled` flags in `config/trainer_config.yaml`:

```yaml
# Enable only fruit detection and defect segmentation
fruit_detection:
  enabled: true
  # ... other parameters

defect_classification:
  enabled: false  # Disable defect classification training
  # ... other parameters

defect_segmentation:
  enabled: true
  # ... other parameters
```

### Example 3: Creating Custom Dataset Configurations

If you need to create dataset configuration files, you can use the `create_dataset_yaml` function:

```python
from src.train_model import create_dataset_yaml

# Create a dataset configuration for fruit detection (3 classes)
create_dataset_yaml(
    dataset_path='/path/to/fruit_dataset',
    output_path='configs/fruit_detection_dataset.yaml',
    num_classes=3,
    class_names=['apple', 'banana', 'tomato']
)

# Create a dataset configuration for defect classification (2 classes)
create_dataset_yaml(
    dataset_path='/path/to/defect_dataset',
    output_path='configs/defect_classification_dataset.yaml',
    num_classes=2,
    class_names=['non_defective', 'defective']
)

# Create a dataset configuration for defect segmentation (defect classes with masks)
create_dataset_yaml(
    dataset_path='/path/to/defect_segmentation_dataset',
    output_path='configs/defect_segmentation_dataset.yaml',
    num_classes=2,  # defective, background
    class_names=['background', 'defect'],
    task_type='segmentation'
)
```

## Customization Guide

### Training for Different Fruit Types

To customize the training for different fruit types:

1. Update your dataset to include the new fruit classes
2. Modify the number of classes (`nc`) and class names (`names`) in your dataset YAML file
3. Adjust the training script if needed to accommodate the new classes

### Training for Different Defect Categories

To customize defect classification:

1. Update your dataset with appropriate defect categories
2. For binary classification (defective/non-defective), keep 2 classes
3. For multi-class defect classification, modify the number of classes accordingly

### Adjusting Training Parameters

Consider these guidelines when adjusting training parameters:

- **Epochs**: More epochs may improve accuracy but increase training time and risk of overfitting
- **Image Size**: Larger images may improve accuracy but require more computational resources
- **Batch Size**: Larger batches may train faster but require more memory
- **Learning Rate**: Higher rates may train faster but could miss optimal solutions; lower rates are more precise but slower

## Model Management

### Pre-trained Model Downloads

The training script automatically downloads the required pre-trained models:

- **Fruit Detection**: `yolov8n.pt` - A detection model pre-trained on COCO dataset
- **Defect Segmentation**: `yolov8n-seg.pt` - A segmentation model pre-trained on COCO dataset
- **Defect Classification**: `yolov8n-cls.pt` - A classification model pre-trained on ImageNet dataset

These models are downloaded automatically by the Ultralytics library when first needed and cached in the system's default location (typically in the user's home directory under `.ultralytics`).

### Model Storage

After training is completed, the final trained models are saved in the appropriate subdirectories within the `models/` directory:

- **Fruit Detection Models**: Saved to `models/fruit_detection/` as `fruit_detector.pt`
- **Defect Classification Models**: Saved to `models/defect_classification/` as `defect_classifier.pt`
- **Defect Segmentation Models**: Saved to `models/defect_segmentation/` as `defect_segmenter.pt`

The training results, logs, and checkpoints during training are saved in the directory specified by the `save_dir` parameter in the configuration file (default: `runs/train`).

### Advanced Training Features

#### Custom Loss Functions
The defect segmentation model supports custom loss functions including:
- **Dice Loss**: For better handling of imbalanced segmentation masks
- **Focal Loss**: For focusing on hard examples during training
- **Combined Losses**: Automatic combination of standard and custom losses

These can be configured in the `defect_segmentation` section of your configuration file:
```yaml
defect_segmentation:
 # ... other parameters
  loss_function: 'auto'  # Options: 'auto', 'dice', 'focal', 'ce' (cross-entropy)
```

#### Environmental Augmentations
The training system includes advanced environmental augmentation capabilities to improve model robustness:
- **Glare simulation**: For handling bright lighting conditions
- **Shadow augmentation**: For varying lighting conditions
- **Blur effects**: To improve robustness to motion blur
- **Low-light simulation**: For poor visibility conditions

These can be enabled/disabled in the configuration:
```yaml
defect_segmentation:
 # ... other parameters
  augment_env_effects: true # Enable environmental augmentations
  auto_augment: 'randaugment'  # Auto augmentation policy
```

#### Advanced Training Parameters
The system supports additional training parameters for fine-tuning:
- **Patience**: Number of epochs to wait before early stopping
- **Warmup epochs**: Number of epochs for learning rate warmup
- **Cosine LR scheduling**: Cosine annealing for learning rate scheduling

Example configuration:
```yaml
defect_segmentation:
  # ... other parameters
  patience: 100
  warmup_epochs: 3
  cosine_lr: true
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Out of Memory Error
**Solution**: Reduce the batch size or image size to decrease memory usage

#### Issue: Slow Training Performance
**Solution**: 
- Ensure CUDA is available and properly configured for GPU training
- Consider reducing image size or increasing batch size for better GPU utilization
- Verify that your dataset paths are correct and accessible

#### Issue: Poor Model Performance
**Solution**:
- Check that your dataset is well-labeled and balanced
- Try adjusting hyperparameters (learning rate, epochs)
- Consider data augmentation techniques
- Verify that your validation set is representative

#### Issue: Dataset Path Errors
**Solution**: 
- Verify that your dataset YAML file points to the correct directories
- Ensure that image and label paths in your dataset are correctly structured
- Check that all required directories and files exist

#### Issue: Model Download Failures
**Solution**:
- Ensure you have an active internet connection
- Check firewall/proxy settings if working in a restricted environment
- Manually download the pre-trained models if needed

### Verifying Dataset Structure

Your dataset should be structured as follows:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

The `dataset.yaml` file should contain:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 3  # number of classes
names: ['apple', 'banana', 'tomato']  # class names
```

### Model Output Locations

Trained models are saved in two locations:

1. **During Training**: Training results, logs, and checkpoints are saved in the directory specified by `--save_dir` (default: `runs/train`)
2. **Final Models**: The final trained models are saved in the `models/` subdirectories:
   - Fruit detection: `models/fruit_detection/fruit_detector.pt`
   - Defect classification: `models/defect_classification/defect_classifier.pt`
   - Defect segmentation: `models/defect_segmentation/defect_segmenter.pt`
### Monitoring Training Progress

Training progress is displayed in the console during execution. Additionally, training metrics and visualizations are saved in the output directory for later analysis.

## Metrics Evaluation

After each model training session, the system automatically performs comprehensive metrics evaluation including:

### Performance Metrics
- Inference time metrics (p50/p90/p99 percentiles) with targets ≤200ms per frame
- Throughput in frames per second (FPS)
- End-to-end latency measurements
- Model loading time

### Classification Metrics
- Overall precision, recall, F1-score, and accuracy
- Per-class metrics for apples, bananas, and tomatoes
- Per-class metrics for defective/non-defective categories

### Hardware Metrics
- GPU/CPU usage percentages
- Memory consumption
- GPU memory usage (if applicable)

### Model Metrics
- Model size in MB
- Loading time
- Architecture efficiency metrics

### Out-of-Distribution (OOD) Detection
- OOD detection rate
- OOD flag analysis
- Robustness testing for unknown objects

### Validation and Stress Testing
- Performance on train/val/test sets
- Stress testing for poor quality images (glare/shadows/blur)
- Calibration metrics
- Performance distribution reports
- KPI validation against targets (recall ≥0.80 for problematic cases, precision ≥0.85 for marked items, ≤10% unnecessary suggestions)
### Output Format
- Metrics are saved to timestamped JSON and CSV files in the `/metrics` folder
- JSON files contain the complete hierarchical structure with nested metrics categories
- CSV files contain a flattened representation where nested metrics are represented with dot notation (e.g., "classification_metrics.overall.precision")
- Both formats include detailed performance distribution reports and hardware usage logs for on-device operation validation

