"""
Training script for the multi-model setup of fruit defect detection system.

This script handles training of:
 1. Fruit detection model (YOLOv8n detection) for detecting fruit types with bounding boxes
 2. Defect classification model for binary classification (defective/non_defective)
 3. Defect segmentation model (YOLOv8n-seg) for segmentation masks of defects

The script is designed to be run independently from the main application and
supports separate training workflows for each model type with appropriate
dataset format handling for both detection (bounding boxes) and segmentation (masks).
"""
import os
import time
from ultralytics import YOLO
import yaml
import sys
import logging

# Add the utils directory to the path to import metrics_calculator and image_utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from metrics_calculator import calculate_comprehensive_metrics
    from image_utils import SegmentationAugmentation, apply_environmental_augmentation
except ImportError as e:
    logging.warning(f"Import error: {e}. Please install required dependencies.")
    calculate_comprehensive_metrics = None
    SegmentationAugmentation = None
    apply_environmental_augmentation = None

def ensure_model_directory(model_type):
    """
    Ensure the appropriate model directory exists and return the path.
    
    Args:
        model_type (str): Type of model ('fruit_detection', 'defect_classification', or 'defect_segmentation')
    
    Returns:
        str: Path to the model directory
    """
    base_model_dir = "models"
    if model_type == 'fruit_detection':
        model_dir = os.path.join(base_model_dir, "fruit_detection")
    elif model_type == 'defect_classification':
        model_dir = os.path.join(base_model_dir, "defect_classification")
    elif model_type == 'defect_segmentation':
        model_dir = os.path.join(base_model_dir, "defect_segmentation")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

from ultralytics.models.yolo.segment import SegmentationTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomSegmentationTrainer(SegmentationTrainer):
    """
    Custom segmentation trainer that supports additional loss functions like Dice and Focal loss,
    and incorporates advanced augmentation strategies for segmentation tasks.
    """
    def __init__(self, loss_function='auto', augment_env_effects=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function
        self.augment_env_effects = augment_env_effects
        
        # Initialize custom loss functions if needed
        if loss_function == 'focal':
            self.criterion_focal = FocalLoss()
        elif loss_function == 'dice':
            self.criterion_dice = DiceLoss()
    
    def criterion(self, preds, batch):
        """
        Custom loss function that can incorporate Dice or Focal loss.
        """
        # Get the original loss from parent class
        loss, batch_size = super().criterion(preds, batch)
        
        # If using custom loss functions, modify the loss accordingly
        if self.loss_function in ['focal', 'dice']:
            # Extract predictions and targets
            batch_idx = batch['batch_idx']
            cls = batch['cls']
            bboxes = batch['bboxes']
            masks = batch.get('masks')  # Segmentation masks
            
            if masks is not None and self.loss_function == 'focal':
                # Apply focal loss to mask predictions if available
                mask_loss_focal = self.criterion_focal(preds[1], masks)  # Assuming preds[1] contains mask predictions
                # Combine with original loss (adjust weights as needed)
                loss = loss + 0.5 * mask_loss_focal
            elif masks is not None and self.loss_function == 'dice':
                # Apply dice loss to mask predictions if available
                mask_loss_dice = self.criterion_dice(preds[1], masks)  # Assuming preds[1] contains mask predictions
                # Combine with original loss (adjust weights as needed)
                loss = loss + 0.5 * mask_loss_dice
        
        return loss, batch_size


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in segmentation tasks.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def download_pretrained_model(model_name, model_dir):
    """
    Download or load a pretrained model, saving it to the specified directory if needed.
    
    Args:
        model_name (str): Name of the model to load (e.g., 'yolov8n.pt', 'yolov8n-seg.pt', 'yolov8n-cls.pt')
        model_dir (str): Directory to save/load the model
    
    Returns:
        YOLO: Loaded model instance
    """
    # Ultralytics will automatically download the model if not present
    # The model will be cached in the ultralytics cache by default
    model = YOLO(model_name)
    
    # Note: Ultralytics automatically handles the download to its own cache
    # We can't directly control where the initial download goes, but we can
    # copy the model to our models directory after training if needed
    
    return model


def train_fruit_detection_model(data_path, epochs=100, img_size=640, batch_size=16,
                               learning_rate=0.01, save_period=10, save_dir='runs/detect/fruit_detector',
                               model_name='yolov8n.pt'):
    """
    Train the fruit detection model using detection format (bounding boxes).
    
    This model is responsible for detecting fruit types (apple, banana, tomato)
    with bounding boxes in the input images. Training time is measured and
    included in the metrics if enabled in the configuration.
    
    Args:
        data_path (str): Path to the dataset YAML file with bounding box annotations in YOLO format
        epochs (int): Number of training epochs
        img_size (int): Training image size
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        save_dir (str): Directory to save the trained model
        model_name (str): Name of the pretrained model to start from
    """
    print("Starting training of fruit detection model...")
    print(f"Dataset path: {data_path}")
    print(f"Training parameters - Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Load a detection model (YOLOv8n for detection, not segmentation)
    # Ultralytics will automatically download the model if not present
    model = YOLO(model_name)  # Using pre-trained detection model as starting point
    
    # Record start time for training duration
    start_time = time.time()
    
    # Train the model using detection format (bounding boxes)
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=learning_rate,
        save_period=save_period,
        save_dir=save_dir,
        name='fruit_detector'
    )
    
    # Calculate training time
    training_duration = time.time() - start_time
    
    # Save the final trained model to the models directory
    model_dir = ensure_model_directory('fruit_detection')
    final_model_path = os.path.join(model_dir, 'fruit_detector.pt')
    model.save(final_model_path)
    print(f"Trained fruit detection model saved to {final_model_path}")
    print(f"Training completed in {training_duration:.2f} seconds")
    print("Fruit detection model training completed!")
    
    # Calculate comprehensive metrics if metrics calculator is available and enabled
    # Load configuration to check master metrics toggle
    config = load_trainer_config()
    if calculate_comprehensive_metrics is not None and config['trainer']['enable_metrics']:
        print("Calculating comprehensive metrics for fruit detection model...")
        # Define paths for test images - these would need to exist in the actual dataset
        # data_path is the YAML file path, so we need to get the directory containing the YAML file
        import yaml
        dataset_dir = os.path.dirname(data_path)
        if not dataset_dir:
            dataset_dir = "."  # Use current directory if no path is specified
        
        # Load the dataset YAML file to get the actual paths
        with open(data_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Get the test images path from the YAML config
        test_path = dataset_config.get('test', 'test')
        # If the path is relative, resolve it relative to the YAML file location
        if test_path.startswith('../') or test_path.startswith('./') or '/' in test_path:
            test_images_path = os.path.join(os.path.dirname(data_path), test_path)
        else:
            test_images_path = os.path.join(dataset_dir, test_path)
        
        # Check if the path exists, and if not try common structures
        if not os.path.exists(test_images_path):
            # Try the common structure of test/images
            test_images_path = os.path.join(dataset_dir, 'test', 'images')
            if not os.path.exists(test_images_path):
                # If that doesn't exist either, default to test directory
                test_images_path = os.path.join(dataset_dir, 'test')
        
        ood_images_path = os.path.join(dataset_dir, 'ood_images') if os.path.exists(os.path.join(dataset_dir, 'ood_images')) else None
        calibration_images_path = os.path.join(dataset_dir, 'calibration_images') if os.path.exists(os.path.join(dataset_dir, 'calibration_images')) else None
        stress_test_images_path = os.path.join(dataset_dir, 'stress_test_images') if os.path.exists(os.path.join(dataset_dir, 'stress_test_images')) else None
        
        # Check if training time calculation is enabled in the configuration
        calculate_training_time = config.get('metrics', {}).get('calculate_training_time', True)
        training_time_param = training_duration if calculate_training_time else None
        
        metrics = calculate_comprehensive_metrics(
            model_path=final_model_path,
            data_path=data_path,
            test_images_path=test_images_path,
            ood_images_path=ood_images_path,
            calibration_images_path=calibration_images_path,
            stress_test_images_path=stress_test_images_path,
            model_config_path="config/model_config.yaml",  # Pass model config for class validation
            training_time=training_time_param  # Pass the training time if enabled in config
        )
        print("Metrics calculation completed!")

    return model


def train_defect_classification_model(data_path, epochs=50, img_size=224, batch_size=32,
                                    learning_rate=0.001, save_period=10, save_dir='runs/classify/defect_classifier',
                                    model_name='yolov8n-cls.pt'):
    """
    Train the defect classification model for binary classification (defective/non_defective).
    
    This model is responsible for determining whether detected fruits or regions contain defects.
    Training time is measured and included and the metrics if enabled in the configuration.
    
    Args:
        data_path (str): Path to the dataset YAML file for defect classification
        epochs (int): Number of training epochs
        img_size (int): Training image size
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        save_dir (str): Directory to save the trained model
        model_name (str): Name of the pretrained model to start from
    """
    print("Starting training of defect classification model...")
    print(f"Dataset path: {data_path}")
    print(f"Training parameters - Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Load a classification model
    # Ultralytics will automatically download the model if not present
    model = YOLO(model_name)  # Using pre-trained classification model as starting point
    
    # Record start time for training duration
    start_time = time.time()
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=learning_rate,
        save_period=save_period,
        save_dir=save_dir,
        name='defect_classifier'
    )
    
    # Calculate training time
    training_duration = time.time() - start_time
    
    # Save the final trained model to the models directory
    model_dir = ensure_model_directory('defect_classification')
    final_model_path = os.path.join(model_dir, 'defect_classifier.pt')
    model.save(final_model_path)
    print(f"Trained defect classification model saved to {final_model_path}")
    print(f"Training completed in {training_duration:.2f} seconds")
    print("Defect classification model training completed!")
    
    # Calculate comprehensive metrics if metrics calculator is available and enabled
    # Load configuration to check master metrics toggle
    config = load_trainer_config()
    if calculate_comprehensive_metrics is not None and config['trainer']['enable_metrics']:
        print("Calculating comprehensive metrics for defect classification model...")
        # Define paths for test images - these would need to exist in the actual dataset
        # data_path is the YAML file path, so we need to get the directory containing the YAML file
        import yaml
        dataset_dir = os.path.dirname(data_path)
        if not dataset_dir:
            dataset_dir = "."  # Use current directory if no path is specified
        
        # Load the dataset YAML file to get the actual paths
        with open(data_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Get the test images path from the YAML config
        test_path = dataset_config.get('test', 'test')
        # If the path is relative, resolve it relative to the YAML file location
        if test_path.startswith('../') or test_path.startswith('./') or '/' in test_path:
            test_images_path = os.path.join(os.path.dirname(data_path), test_path)
        else:
            test_images_path = os.path.join(dataset_dir, test_path)
        
        # Check if the path exists, and if not try common structures
        if not os.path.exists(test_images_path):
            # Try the common structure of test/images
            test_images_path = os.path.join(dataset_dir, 'test', 'images')
            if not os.path.exists(test_images_path):
                # If that doesn't exist either, default to test directory
                test_images_path = os.path.join(dataset_dir, 'test')
        
        ood_images_path = os.path.join(dataset_dir, 'ood_images') if os.path.exists(os.path.join(dataset_dir, 'ood_images')) else None
        calibration_images_path = os.path.join(dataset_dir, 'calibration_images') if os.path.exists(os.path.join(dataset_dir, 'calibration_images')) else None
        stress_test_images_path = os.path.join(dataset_dir, 'stress_test_images') if os.path.exists(os.path.join(dataset_dir, 'stress_test_images')) else None
        
        # Check if training time calculation is enabled in the configuration
        calculate_training_time = config.get('metrics', {}).get('calculate_training_time', True)
        training_time_param = training_duration if calculate_training_time else None
        
        metrics = calculate_comprehensive_metrics(
            model_path=final_model_path,
            data_path=data_path,
            test_images_path=test_images_path,
            ood_images_path=ood_images_path,
            calibration_images_path=calibration_images_path,
            stress_test_images_path=stress_test_images_path,
            model_config_path="config/model_config.yaml", # Pass model config for class validation
            training_time=training_time_param  # Pass the training time if enabled in config
        )
        print("Metrics calculation completed!")

    return model


def train_defect_segmentation_model(data_path, epochs=150, img_size=640, batch_size=8,
                                   learning_rate=0.001, save_period=10, save_dir='runs/segment/defect_segmenter',
                                   loss_function='auto', patience=100, warmup_epochs=3, cosine_lr=True,
                                   augment_env_effects=True, auto_augment='randaugment',
                                   model_name='yolov8n-seg.pt'):
    """
    Train the defect segmentation model using YOLOv8 segmentation format.
    
    This model is responsible for creating precise segmentation masks for defects
    on detected fruits, providing pixel-level accuracy for defect regions.
    Training time is measured and included in the metrics if enabled in the configuration.
    
    Args:
        data_path (str): Path to the dataset YAML file with segmentation mask annotations in YOLO format
        epochs (int): Number of training epochs
        img_size (int): Training image size
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        save_dir (str): Directory to save the trained model
        loss_function (str): Loss function to use ('auto', 'dice', 'focal', 'ce' for cross-entropy)
        patience (int): Number of epochs to wait without improvement before early stopping
        warmup_epochs (int): Number of warmup epochs
        cosine_lr (bool): Whether to use cosine learning rate scheduling
        augment_env_effects (bool): Whether to apply environmental augmentations (glare, shadows, blur)
        auto_augment (str): Auto augmentation policy (e.g., 'randaugment', 'autoaugment')
        model_name (str): Name of the pretrained model to start from
    """
    print("Starting training of defect segmentation model...")
    print(f"Dataset path: {data_path}")
    print(f"Training parameters - Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Loss function: {loss_function}, Patience: {patience}, Warmup epochs: {warmup_epochs}, Cosine LR: {cosine_lr}")
    print(f"Environmental augmentations: {augment_env_effects}, Auto augment: {auto_augment}")
    
    # Load a segmentation model
    # Ultralytics will automatically download the model if not present
    model = YOLO(model_name) # Using pre-trained segmentation model as starting point
    
    # Record start time for training duration
    start_time = time.time()
    
    # For custom loss functions, we need to use a custom trainer
    if loss_function in ['focal', 'dice'] or augment_env_effects:
        # Update the model's trainer to use our custom implementation
        from ultralytics.cfg import get_cfg
        args = dict(
            model=model_name,
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            lr0=learning_rate,
            save_period=save_period,
            save_dir=save_dir,
            name='defect_segmenter',
            patience=patience,
            warmup_epochs=warmup_epochs,
            auto_augment=auto_augment if auto_augment else None
        )
        
        # Set up the configuration
        cfg = get_cfg()
        for k, v in args.items():
            setattr(cfg, k, v)
        
        # Create custom trainer instance with environmental augmentation support
        trainer = CustomSegmentationTrainer(
            loss_function=loss_function,
            augment_env_effects=augment_env_effects,
            cfg=cfg
        )
        
        # Set the data and build the model
        trainer.setup_model()
        trainer.train()
        
        # Get the trained model from the trainer
        trained_model = trainer.model
    else:
        # Prepare additional training arguments based on loss function and other parameters
        train_kwargs = {
            'data': data_path,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'lr0': learning_rate,
            'save_period': save_period,
            'save_dir': save_dir,
            'name': 'defect_segmenter',
            'patience': patience,
            'warmup_epochs': warmup_epochs,
            'auto_augment': auto_augment if auto_augment else None
        }
        
        # Add loss function specific parameters if needed
        # Note: The actual loss function configuration may need to be handled differently depending on the YOLO version
        # For now, we'll use the parameters that affect loss behavior
        if loss_function == 'focal':
            # Focal loss is not directly available in YOLOv8, but we can adjust parameters that affect loss behavior
            train_kwargs['close_mosaic'] = 10  # Recommended for better loss behavior
            train_kwargs['label_smoothing'] = 0.1  # Helps with focal loss-like behavior
            # Additional parameters that may help with segmentation loss
            train_kwargs['box'] = 7.5  # Box loss gain
            train_kwargs['cls'] = 0.5  # Classification loss gain
            train_kwargs['dfl'] = 1.5  # DFL loss gain
            train_kwargs['epochs'] = epochs  # Ensure epochs is set
            # Segmentation-specific parameters
            train_kwargs['mask_ratio'] = 4  # Ratio of mask to box loss
            train_kwargs['overlap_mask'] = True # Whether masks should overlap
        elif loss_function == 'dice':
            # Dice loss is not directly available in YOLOv8, but we can adjust parameters that affect loss behavior
            train_kwargs['close_mosaic'] = 10  # Recommended for better loss behavior
            train_kwargs['label_smoothing'] = 0.05  # Moderate label smoothing
            # Additional parameters that may help with segmentation loss
            train_kwargs['box'] = 7.5  # Box loss gain
            train_kwargs['cls'] = 0.5  # Classification loss gain
            train_kwargs['dfl'] = 1.5  # DFL loss gain
            train_kwargs['epochs'] = epochs  # Ensure epochs is set
            # Segmentation-specific parameters
            train_kwargs['mask_ratio'] = 4  # Ratio of mask to box loss
            train_kwargs['overlap_mask'] = True # Whether masks should overlap
        elif loss_function == 'auto':
            # Default behavior, no specific parameters needed
            # Add default segmentation-specific parameters
            train_kwargs['mask_ratio'] = 4  # Ratio of mask to box loss
            train_kwargs['overlap_mask'] = True # Whether masks should overlap
        
        # Use cosine learning rate if specified
        if cosine_lr:
            train_kwargs['lrf'] = 0.01  # Final learning rate factor for cosine scheduling
        
        # Train the model using segmentation format (masks)
        results = model.train(**train_kwargs)
    
    # Calculate training time
    training_duration = time.time() - start_time
    
    # Save the final trained model to the models directory
    model_dir = ensure_model_directory('defect_segmentation')
    final_model_path = os.path.join(model_dir, 'defect_segmenter.pt')
    
    # Save the model - use the appropriate model object depending on which trainer was used
    if 'trained_model' in locals():
        # Custom trainer was used, need to save the model from the trainer
        # The trainer.model is the trained PyTorch model, but we need to save it properly
        # Create a YOLO object and update its model with the trained one
        yolo_model = YOLO(model_name)
        yolo_model.model = trained_model.to('cpu')  # Move to CPU before saving
        yolo_model.save(final_model_path)
    else:
        # Standard trainer was used, 'model' is still available
        model.save(final_model_path)
    print(f"Trained defect segmentation model saved to {final_model_path}")
    print(f"Training completed in {training_duration:.2f} seconds")
    print("Defect segmentation model training completed!")
    
    # Calculate comprehensive metrics if metrics calculator is available and enabled
    # Load configuration to check master metrics toggle
    config = load_trainer_config()
    if calculate_comprehensive_metrics is not None and config['trainer']['enable_metrics']:
        print("Calculating comprehensive metrics for defect segmentation model...")
        # Define paths for test images - these would need to exist in the actual dataset
        # data_path is the YAML file path, so we need to get the directory containing the YAML file
        import yaml
        dataset_dir = os.path.dirname(data_path)
        if not dataset_dir:
            dataset_dir = "."  # Use current directory if no path is specified
        
        # Load the dataset YAML file to get the actual paths
        with open(data_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Get the test images path from the YAML config
        test_path = dataset_config.get('test', 'test')
        # If the path is relative, resolve it relative to the YAML file location
        if test_path.startswith('../') or test_path.startswith('./') or '/' in test_path:
            test_images_path = os.path.join(os.path.dirname(data_path), test_path)
        else:
            test_images_path = os.path.join(dataset_dir, test_path)
        
        # Check if the path exists, and if not try common structures
        if not os.path.exists(test_images_path):
            # Try the common structure of test/images
            test_images_path = os.path.join(dataset_dir, 'test', 'images')
            if not os.path.exists(test_images_path):
                # If that doesn't exist either, default to test directory
                test_images_path = os.path.join(dataset_dir, 'test')
        
        ood_images_path = os.path.join(dataset_dir, 'ood_images') if os.path.exists(os.path.join(dataset_dir, 'ood_images')) else None
        calibration_images_path = os.path.join(dataset_dir, 'calibration_images') if os.path.exists(os.path.join(dataset_dir, 'calibration_images')) else None
        stress_test_images_path = os.path.join(dataset_dir, 'stress_test_images') if os.path.exists(os.path.join(dataset_dir, 'stress_test_images')) else None
        
        # Check if training time calculation is enabled in the configuration
        calculate_training_time = config.get('metrics', {}).get('calculate_training_time', True)
        training_time_param = training_duration if calculate_training_time else None
        
        metrics = calculate_comprehensive_metrics(
            model_path=final_model_path,
            data_path=data_path,
            test_images_path=test_images_path,
            ood_images_path=ood_images_path,
            calibration_images_path=calibration_images_path,
            stress_test_images_path=stress_test_images_path,
            model_config_path="config/model_config.yaml", # Pass model config for class validation
            training_time=training_time_param  # Pass the training time if enabled in config
        )
        print("Metrics calculation completed!")

    # Return the appropriate model object
    if 'trained_model' in locals():
        # For custom trainer, we need to return a YOLO object with the trained model
        yolo_model = YOLO(model_name)
        yolo_model.model = trained_model.to('cpu')
        return yolo_model
    else:
        # Standard trainer, return the original model
        return model

def create_dataset_yaml(dataset_path, output_path, num_classes, class_names, task_type='detection'):
    """
    Create a YAML configuration file for the dataset.
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_path (str): Path to save the YAML file
        num_classes (int): Number of classes
        class_names (list): List of class names
        task_type (str): Type of task ('detection', 'classification', or 'segmentation')
    """
    dataset_config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': num_classes,
        'names': class_names
    }
    
    # Add task-specific configurations
    if task_type == 'segmentation':
        dataset_config['train'] = 'images/train'
        dataset_config['val'] = 'images/val'
        dataset_config['test'] = 'images/test'
        # Segmentation datasets have both images and masks
        dataset_config['train_img'] = 'images/train'
        dataset_config['val_img'] = 'images/val'
        dataset_config['train_seg'] = 'masks/train'  # Segmentation masks for training
        dataset_config['val_seg'] = 'masks/val'      # Segmentation masks for validation
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to {output_path}")
    print(f"Task type: {task_type}, Number of classes: {num_classes}, Classes: {class_names}")


def derive_dataset_paths(dataset_yaml_path):
    """
    Automatically derive train, validation, and test paths from the dataset YAML file location.
    
    Args:
        dataset_yaml_path (str): Path to the dataset YAML file
        
    Returns:
        dict: Dictionary containing derived paths for train, val, and test
    """
    # Get the directory containing the dataset YAML file
    dataset_dir = os.path.dirname(dataset_yaml_path)
    
    # If dataset_yaml_path is just a filename (no directory), use current directory
    if not dataset_dir:
        dataset_dir = "."
    
    # Derive paths assuming standard structure:
    # dataset_dir/
    #   images/
    #     train/
    #     val/
    #     test/
    #   labels/ (for detection/segmentation) or annotations/ (for classification)
    #     train/
    #     val/
    #     test/
    #   data.yaml
    
    derived_paths = {
        'train_path': os.path.join(dataset_dir, 'images', 'train'),
        'val_path': os.path.join(dataset_dir, 'images', 'val'),
        'test_path': os.path.join(dataset_dir, 'images', 'test')
    }
    
    return derived_paths


def load_trainer_config(config_path="config/trainer_config.yaml"):
    """
    Load trainer configuration from YAML file.
    
    Args:
        config_path (str): Path to the trainer configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_all_models(config_path="config/trainer_config.yaml"):
    """
    Train all models based on the configuration file.
    
    Args:
        config_path (str): Path to the trainer configuration file
    """
    # Load configuration
    config = load_trainer_config(config_path)
    
    # Train models based on configuration
    if config['fruit_detection']['enabled']:
        print("Starting fruit detection model training...")
        model = train_fruit_detection_model(
            data_path=config['fruit_detection']['dataset_path'],
            epochs=config['fruit_detection']['epochs'],
            img_size=config['fruit_detection']['img_size'],
            batch_size=config['fruit_detection']['batch_size'],
            learning_rate=config['fruit_detection']['learning_rate'],
            save_period=config['fruit_detection']['save_period'],
            save_dir=config['fruit_detection']['save_dir'],
            model_name=config['fruit_detection'].get('model_name', 'yolov8n.pt')
        )
        print("Fruit detection model training completed!")
    
    if config['defect_classification']['enabled']:
        print("Starting defect classification model training...")
        model = train_defect_classification_model(
            data_path=config['defect_classification']['dataset_path'],
            epochs=config['defect_classification']['epochs'],
            img_size=config['defect_classification']['img_size'],
            batch_size=config['defect_classification']['batch_size'],
            learning_rate=config['defect_classification']['learning_rate'],
            save_period=config['defect_classification']['save_period'],
            save_dir=config['defect_classification']['save_dir'],
            model_name=config['defect_classification'].get('model_name', 'yolov8n-cls.pt')
        )
        print("Defect classification model training completed!")
    
    if config['defect_segmentation']['enabled']:
        print("Starting defect segmentation model training...")
        model = train_defect_segmentation_model(
            data_path=config['defect_segmentation']['dataset_path'],
            epochs=config['defect_segmentation']['epochs'],
            img_size=config['defect_segmentation']['img_size'],
            batch_size=config['defect_segmentation']['batch_size'],
            learning_rate=config['defect_segmentation']['learning_rate'],
            save_period=config['defect_segmentation']['save_period'],
            save_dir=config['defect_segmentation']['save_dir'],
            loss_function=config['defect_segmentation'].get('loss_function', 'auto'),
            patience=config['defect_segmentation'].get('patience', 10),
            warmup_epochs=config['defect_segmentation'].get('warmup_epochs', 3),
            cosine_lr=config['defect_segmentation'].get('cosine_lr', True),
            augment_env_effects=config['defect_segmentation'].get('augment_env_effects', True),
            auto_augment=config['defect_segmentation'].get('auto_augment', 'randaugment'),
            model_name=config['defect_segmentation'].get('model_name', 'yolov8n-seg.pt')
        )
        print("Defect segmentation model training completed!")
    
    print("All model training completed!")


def train_single_model(model_type, config_path="config/trainer_config.yaml"):
    """
    Train a single model based on the configuration file.
    
    Args:
        model_type (str): Type of model to train ('fruit_detection', 'defect_classification', 'defect_segmentation')
        config_path (str): Path to the trainer configuration file
    """
    # Load configuration
    config = load_trainer_config(config_path)
    
    if model_type == 'fruit_detection':
        if config['fruit_detection']['enabled']:
            model = train_fruit_detection_model(
                data_path=config['fruit_detection']['dataset_path'],
                epochs=config['fruit_detection']['epochs'],
                img_size=config['fruit_detection']['img_size'],
                batch_size=config['fruit_detection']['batch_size'],
                learning_rate=config['fruit_detection']['learning_rate'],
                save_period=config['fruit_detection']['save_period'],
                save_dir=config['fruit_detection']['save_dir'],
                model_name=config['fruit_detection'].get('model_name', 'yolov8n.pt')
            )
            print(f"Fruit detection model training completed! Model saved in {config['fruit_detection']['save_dir']}")
        else:
            print("Fruit detection model training is disabled in the configuration.")
    
    elif model_type == 'defect_classification':
        if config['defect_classification']['enabled']:
            model = train_defect_classification_model(
                data_path=config['defect_classification']['dataset_path'],
                epochs=config['defect_classification']['epochs'],
                img_size=config['defect_classification']['img_size'],
                batch_size=config['defect_classification']['batch_size'],
                learning_rate=config['defect_classification']['learning_rate'],
                save_period=config['defect_classification']['save_period'],
                save_dir=config['defect_classification']['save_dir'],
                model_name=config['defect_classification'].get('model_name', 'yolov8n-cls.pt')
            )
            print(f"Defect classification model training completed! Model saved in {config['defect_classification']['save_dir']}")
        else:
            print("Defect classification model training is disabled in the configuration.")
    
    elif model_type == 'defect_segmentation':
        if config['defect_segmentation']['enabled']:
            model = train_defect_segmentation_model(
                data_path=config['defect_segmentation']['dataset_path'],
                epochs=config['defect_segmentation']['epochs'],
                img_size=config['defect_segmentation']['img_size'],
                batch_size=config['defect_segmentation']['batch_size'],
                learning_rate=config['defect_segmentation']['learning_rate'],
                save_period=config['defect_segmentation']['save_period'],
                save_dir=config['defect_segmentation']['save_dir'],
                loss_function=config['defect_segmentation'].get('loss_function', 'auto'),
                patience=config['defect_segmentation'].get('patience', 10),
                warmup_epochs=config['defect_segmentation'].get('warmup_epochs', 3),
                cosine_lr=config['defect_segmentation'].get('cosine_lr', True),
                augment_env_effects=config['defect_segmentation'].get('augment_env_effects', True),
                auto_augment=config['defect_segmentation'].get('auto_augment', 'randaugment'),
                model_name=config['defect_segmentation'].get('model_name', 'yolov8n-seg.pt')
            )
            print(f"Defect segmentation model training completed! Model saved in {config['defect_segmentation']['save_dir']}")
        else:
            print("Defect segmentation model training is disabled in the configuration.")


def main(config_path="config/trainer_config.yaml"):
    """
    Main function to train models based on configuration file.
    
    Args:
        config_path (str): Path to the trainer configuration file
    """
    print("Starting training based on configuration file...")
    train_all_models(config_path)
    print("Training process completed!")

def show_help():
    """
    Display help information for the training script.
    """
    help_text = """
Fruit Defect Detection System - Model Training Script

This script trains models for the fruit defect detection system using a comprehensive
YAML-based configuration system. All training parameters are defined in
config/trainer_config.yaml.

Usage:
    python src/train_model.py [options]

Options:
    --help, -h              Show this help message and exit
    --model_type TYPE       Train a specific model type:
                              fruit_detection    - Train fruit detection model
                              defect_classification - Train defect classification model
                              defect_segmentation   - Train defect segmentation model
    --config PATH           Path to trainer configuration file (default: config/trainer_config.yaml)
    --export, -e            Export a trained model to different format:
                              Usage: --export <model_path> <format>
                              Formats: torchscript, onnx, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle

Examples:
    # Train all enabled models according to configuration
    python src/train_model.py
    
    # Train only the fruit detection model
    python src/train_model.py --model_type fruit_detection
    
    # Train with custom configuration file
    python src/train_model.py --config /path/to/custom_config.yaml
    
    # Export a model to ONNX format
    python src/train_model.py --export models/defect_segmentation/defect_segmenter.pt onnx

For comprehensive configuration details, see the trainer documentation file (docs/TRAINER.md).
"""
    print(help_text)


import sys

def show_help():
    """
    Display help information for the training script.
    """
    help_text = """
Fruit Defect Detection System - Model Training Script

This script trains models for the fruit defect detection system using a comprehensive
YAML-based configuration system. All training parameters are defined in
config/trainer_config.yaml.

Usage:
    python src/train_model.py [options]

Options:
    --help, -h              Show this help message and exit
    --model_type TYPE       Train a specific model type:
                              fruit_detection    - Train fruit detection model
                              defect_classification - Train defect classification model
                              defect_segmentation   - Train defect segmentation model
    --config PATH           Path to trainer configuration file (default: config/trainer_config.yaml)

Examples:
    # Train all enabled models according to configuration
    python src/train_model.py
    
    # Train only the fruit detection model
    python src/train_model.py --model_type fruit_detection
    
    # Train with custom configuration file
    python src/train_model.py --config /path/to/custom_config.yaml

For comprehensive configuration details, see the trainer documentation file (docs/TRAINER.md).
"""
    print(help_text)

def main():
    """
    Main function to train models based on configuration file.
    """
    # Check for help flags first
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        return
    
    # Default configuration path
    config_path = "config/trainer_config.yaml"
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--config':
            if i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --config requires a path argument")
                return
        elif sys.argv[i] == '--model_type':
            if i + 1 < len(sys.argv):
                model_type = sys.argv[i + 1]
                print(f"Starting training for model type: {model_type}")
                train_single_model(model_type, config_path)
                print("Training process completed!")
                return
            else:
                print("Error: --model_type requires a model type argument")
                return
        else:
            i += 1
    
    print("Starting training based on configuration file...")
    train_all_models(config_path)
    print("Training process completed!")


def export_model(model_path, export_format='torchscript', output_path=None):
    """
    Export a trained model to different formats for deployment.
    
    Args:
        model_path (str): Path to the trained model (.pt file)
        export_format (str): Format to export to ('torchscript', 'onnx', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle')
        output_path (str): Path to save the exported model (optional, defaults to model path with new extension)
    
    Returns:
        str: Path to the exported model
    """
    from ultralytics import YOLO
    import os
    
    print(f"Exporting model from {model_path} to {export_format} format...")
    
    # Load the model
    model = YOLO(model_path)
    
    # Determine output path
    if output_path is None:
        base_path = os.path.splitext(model_path)[0]
        format_extension = {
            'torchscript': '.torchscript',
            'onnx': '.onnx',
            'engine': '.engine',
            'coreml': '.mlmodel',
            'saved_model': '_saved_model',
            'pb': '.pb',
            'tflite': '.tflite',
            'edgetpu': '_edgetpu.tflite',
            'tfjs': '_web_model',
            'paddle': '.pdmodel'
        }
        ext = format_extension.get(export_format, f'.{export_format}')
        if export_format == 'saved_model':  # Special case for directory output
            output_path = f"{base_path}{ext}"
        else:
            output_path = f"{base_path}{ext}"
    
    # Export the model
    success = model.export(format=export_format, save_dir=os.path.dirname(output_path), name=os.path.basename(output_path))
    
    print(f"Model exported successfully to {output_path}")
    return output_path


def main():
    """
    Main function to train models based on configuration file.
    """
    # Check for help flags first
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        return
    
    # Check for export functionality
    if '--export' in sys.argv or '-e' in sys.argv:
        try:
            model_idx = sys.argv.index('--export') if '--export' in sys.argv else sys.argv.index('-e')
            if model_idx + 2 < len(sys.argv):
                model_path = sys.argv[model_idx + 1]
                export_format = sys.argv[model_idx + 2]
                export_model(model_path, export_format)
                print("Model export completed!")
            else:
                print("Error: --export requires model_path and export_format arguments")
                print("Usage: python train_model.py --export <model_path> <format>")
        except (ValueError, IndexError):
            print("Error: --export requires model_path and export_format arguments")
            print("Usage: python train_model.py --export <model_path> <format>")
        return
    
    # Default configuration path
    config_path = "config/trainer_config.yaml"
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--config':
            if i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --config requires a path argument")
                return
        elif sys.argv[i] == '--model_type':
            if i + 1 < len(sys.argv):
                model_type = sys.argv[i + 1]
                print(f"Starting training for model type: {model_type}")
                train_single_model(model_type, config_path)
                print("Training process completed!")
                return
            else:
                print("Error: --model_type requires a model type argument")
                return
        else:
            i += 1
    
    print("Starting training based on configuration file...")
    train_all_models(config_path)
    print("Training process completed!")


if __name__ == "__main__":
    main()
