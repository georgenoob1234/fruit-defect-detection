# Training Script Documentation

## Overview

This documentation covers the training script `src/train_model.py`, which is designed for training the multi-model setup of the fruit defect detection system. The script handles training of the fruit detection model, defect classification model, and defect segmentation model with separate workflows for each model type.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation Requirements](#installation-requirements)
3. [Usage Instructions](#usage-instructions)
4. [Command-Line Arguments](#command-line-arguments)
5. [Configuration Options](#configuration-options)
6. [Dual-Model Training Process](#dual-model-training-process)
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

The training script can be executed from the command line with various parameters:

```bash
python src/train_model.py --model_type <model_type> --data_path <dataset_path> [options]
```

### Basic Usage

To train the fruit detection model:
```bash
python src/train_model.py --model_type fruit_detection --data_path /path/to/dataset.yaml
```

To train the defect classification model:
```bash
python src/train_model.py --model_type defect_classification --data_path /path/to/dataset.yaml
```

## Command-Line Arguments

The script accepts the following command-line arguments:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model_type` | string | Yes | N/A | Type of model to train. Options: `fruit_detection` (bounding boxes), `defect_classification` (binary classification), `defect_segmentation` (segmentation masks) |
| `--data_path` | string | Yes | N/A | Path to the dataset YAML file or directory |
| `--epochs` | int | No | 100 | Number of training epochs |
| `--img_size` | int | No | 320 | Training image size (both width and height) |
| `--batch_size` | int | No | 16 | Training batch size |
| `--learning_rate` | float | No | 0.01 | Learning rate for training |
| `--save_dir` | string | No | Based on model type | Directory to save the training results and checkpoints |

## Configuration Options

### Fruit Detection Model Configuration

When `--model_type` is set to `fruit_detection`, the script trains a YOLOv8n-seg model with the following characteristics:

- **Model Type**: YOLOv8n-seg (segmentation model)
- **Classes**: Expected to be 3 classes (apple, banana, tomato)
- **Output**: Bounding boxes and segmentation masks for detected fruits
- **Purpose**: Detect fruit types and propose regions where defects might be located


### Defect Segmentation Model Configuration

When `--model_type` is set to `defect_segmentation`, the script trains a YOLOv8n-seg model with the following characteristics:

- **Model Type**: YOLOv8n-seg (segmentation model)
- **Classes**: Expected to be defect classes with segmentation masks
- **Output**: Pixel-level segmentation masks for defects on detected fruits
- **Purpose**: Create precise segmentation masks for defects, providing pixel-level accuracy for defect regions

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Number of complete passes through the training dataset |
| `img_size` | 320 | Size of input images (squared: 320x320) |
| `batch_size` | 16 | Number of samples processed together in each training step |
| `learning_rate` | 0.01 | Rate at which the model learns from the data |
| `save_dir` | runs/train | Directory where training results, checkpoints and logs are saved |

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

### Example 1: Training the Fruit Detection Model with Custom Parameters

```bash
python src/train_model.py \
    --model_type fruit_detection \
    --data_path datasets/fruit_dataset.yaml \
    --epochs 200 \
    --img_size 384 \
    --batch_size 8 \
    --learning_rate 0.005 \
    --save_dir runs/train/fruit_detector
```


### Example 3: Training the Defect Segmentation Model with Custom Parameters

```bash
python src/train_model.py \
    --model_type defect_segmentation \
    --data_path datasets/defect_segmentation_dataset.yaml \
    --epochs 150 \
    --img_size 640 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --save_dir runs/segment/defect_segmenter
```

To train the defect segmentation model:
```bash
python src/train_model.py --model_type defect_segmentation --data_path /path/to/dataset.yaml
```

### Example 3: Using the Dataset YAML Creator Function

If you need to create a dataset configuration file, you can modify the script to use the `create_dataset_yaml` function:

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
```

# Create a dataset configuration for defect segmentation (defect classes with masks)
create_dataset_yaml(
    dataset_path='/path/to/defect_segmentation_dataset',
    output_path='configs/defect_segmentation_dataset.yaml',
    num_classes=2,  # defective, background
    class_names=['background', 'defect'],
    task_type='segmentation'
)

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

These models are downloaded automatically by the Ultralytics library when first needed and cached in the system's default location (typically in the user's home directory under `.ultralytics`).

### Model Storage

After training is completed, the final trained models are saved in the appropriate subdirectories within the `models/` directory:

- **Fruit Detection Models**: Saved to `models/fruit_detection/` as `fruit_detector.pt`
- **Defect Segmentation Models**: Saved to `models/defect_segmentation/` as `defect_segmenter.pt`

The training results, logs, and checkpoints during training are saved in the directory specified by the `--save_dir` parameter (default: `runs/train`).

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