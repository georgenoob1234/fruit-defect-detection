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
import argparse
import os
from ultralytics import YOLO
import yaml


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
                               learning_rate=0.01, save_dir='runs/detect/fruit_detector'):
    """
    Train the fruit detection model using detection format (bounding boxes).
    
    This model is responsible for detecting fruit types (apple, banana, tomato)
    with bounding boxes in the input images.
    
    Args:
        data_path (str): Path to the dataset YAML file with bounding box annotations in YOLO format
        epochs (int): Number of training epochs
        img_size (int): Training image size
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        save_dir (str): Directory to save the trained model
    """
    print("Starting training of fruit detection model...")
    print(f"Dataset path: {data_path}")
    print(f"Training parameters - Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Load a detection model (YOLOv8n for detection, not segmentation)
    # Ultralytics will automatically download yolov8n.pt if not present
    model = YOLO('yolov8n.pt')  # Using pre-trained detection model as starting point
    
    # Train the model using detection format (bounding boxes)
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=learning_rate,
        save_dir=save_dir,
        name='fruit_detector'
    )
    
    # Save the final trained model to the models directory
    model_dir = ensure_model_directory('fruit_detection')
    final_model_path = os.path.join(model_dir, 'fruit_detector.pt')
    model.save(final_model_path)
    print(f"Trained fruit detection model saved to {final_model_path}")
    
    print("Fruit detection model training completed!")
    return model


def train_defect_classification_model(data_path, epochs=50, img_size=224, batch_size=32,
                                    learning_rate=0.001, save_dir='runs/classify/defect_classifier'):
    """
    Train the defect classification model for binary classification (defective/non_defective).
    
    This model is responsible for determining whether detected fruits or regions contain defects.
    
    Args:
        data_path (str): Path to the dataset YAML file for defect classification
        epochs (int): Number of training epochs
        img_size (int): Training image size
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        save_dir (str): Directory to save the trained model
    """
    print("Starting training of defect classification model...")
    print(f"Dataset path: {data_path}")
    print(f"Training parameters - Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Load a classification model (YOLOv8n-cls)
    # Ultralytics will automatically download yolov8n-cls.pt if not present
    model = YOLO('yolov8n-cls.pt')  # Using pre-trained classification model as starting point
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=learning_rate,
        save_dir=save_dir,
        name='defect_classifier'
    )
    
    # Save the final trained model to the models directory
    model_dir = ensure_model_directory('defect_classification')
    final_model_path = os.path.join(model_dir, 'defect_classifier.pt')
    model.save(final_model_path)
    print(f"Trained defect classification model saved to {final_model_path}")
    
    print("Defect classification model training completed!")
    return model


def train_defect_segmentation_model(data_path, epochs=150, img_size=640, batch_size=8,
                                   learning_rate=0.001, save_dir='runs/segment/defect_segmenter'):
    """
    Train the defect segmentation model using YOLOv8 segmentation format.
    
    This model is responsible for creating precise segmentation masks for defects
    on detected fruits, providing pixel-level accuracy for defect regions.
    
    Args:
        data_path (str): Path to the dataset YAML file with segmentation mask annotations in YOLO format
        epochs (int): Number of training epochs
        img_size (int): Training image size
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        save_dir (str): Directory to save the trained model
    """
    print("Starting training of defect segmentation model...")
    print(f"Dataset path: {data_path}")
    print(f"Training parameters - Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Load a segmentation model (YOLOv8n-seg for segmentation tasks)
    # Ultralytics will automatically download yolov8n-seg.pt if not present
    model = YOLO('yolov8n-seg.pt')  # Using pre-trained segmentation model as starting point
    
    # Train the model using segmentation format (masks)
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=learning_rate,
        save_dir=save_dir,
        name='defect_segmenter'
    )
    
    # Save the final trained model to the models directory
    model_dir = ensure_model_directory('defect_segmentation')
    final_model_path = os.path.join(model_dir, 'defect_segmenter.pt')
    model.save(final_model_path)
    print(f"Trained defect segmentation model saved to {final_model_path}")
    
    print("Defect segmentation model training completed!")
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


def main():
    parser = argparse.ArgumentParser(description='Train models for fruit defect detection system')
    parser.add_argument('--model_type', type=str, choices=['fruit_detection', 'defect_classification', 'defect_segmentation'],
                       required=True, help='Type of model to train: fruit_detection (bounding boxes), '
                                          'defect_classification (binary classification), '
                                          'or defect_segmentation (segmentation masks)')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to the dataset YAML file or directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=320, help='Training image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--save_dir', type=str, help='Directory to save the model')
    
    args = parser.parse_args()
    
    # Ensure model directory exists
    model_dir = ensure_model_directory(args.model_type)
    
    # Set default save directory if not provided
    if not args.save_dir:
        if args.model_type == 'fruit_detection':
            args.save_dir = 'runs/detect/fruit_detector'
        elif args.model_type == 'defect_classification':
            args.save_dir = 'runs/classify/defect_classifier'
        elif args.model_type == 'defect_segmentation':
            args.save_dir = 'runs/segment/defect_segmenter'
    
    if args.model_type == 'fruit_detection':
        # For fruit detection, we expect 3 classes: apple, banana, tomato
        # Using detection format with appropriate image size
        detection_img_size = args.img_size  # Use the specified image size
        model = train_fruit_detection_model(
            data_path=args.data_path,
            epochs=args.epochs,
            img_size=detection_img_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir
        )
    elif args.model_type == 'defect_classification':
        # For defect classification, we expect 2 classes: defective, non_defective
        model = train_defect_classification_model(
            data_path=args.data_path,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir
        )
    elif args.model_type == 'defect_segmentation':
        # For defect segmentation, using YOLOv8 segmentation format for defect masks
        seg_img_size = args.img_size  # Use the specified image size
        model = train_defect_segmentation_model(
            data_path=args.data_path,
            epochs=args.epochs,
            img_size=seg_img_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir
        )
    
    print(f"Model training completed. Model saved in {args.save_dir}")
    print(f"Final trained model also saved in {model_dir}")


if __name__ == "__main__":
    main()