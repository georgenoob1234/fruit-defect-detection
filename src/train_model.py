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
from ultralytics import YOLO
import yaml
import sys
import logging

# Add the utils directory to the path to import metrics_calculator
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from metrics_calculator import calculate_comprehensive_metrics
except ImportError:
    logging.warning("Metrics calculator module not found. Please install required dependencies.")
    calculate_comprehensive_metrics = None

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
        
        metrics = calculate_comprehensive_metrics(
            model_path=final_model_path,
            data_path=data_path,
            test_images_path=test_images_path,
            ood_images_path=ood_images_path,
            calibration_images_path=calibration_images_path,
            stress_test_images_path=stress_test_images_path,
            model_config_path="config/model_config.yaml"  # Pass model config for class validation
        )
        print("Metrics calculation completed!")

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
        
        metrics = calculate_comprehensive_metrics(
            model_path=final_model_path,
            data_path=data_path,
            test_images_path=test_images_path,
            ood_images_path=ood_images_path,
            calibration_images_path=calibration_images_path,
            stress_test_images_path=stress_test_images_path,
            model_config_path="config/model_config.yaml" # Pass model config for class validation
        )
        print("Metrics calculation completed!")

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
        
        metrics = calculate_comprehensive_metrics(
            model_path=final_model_path,
            data_path=data_path,
            test_images_path=test_images_path,
            ood_images_path=ood_images_path,
            calibration_images_path=calibration_images_path,
            stress_test_images_path=stress_test_images_path,
            model_config_path="config/model_config.yaml" # Pass model config for class validation
        )
        print("Metrics calculation completed!")

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
            save_dir=config['fruit_detection']['save_dir']
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
            save_dir=config['defect_classification']['save_dir']
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
            save_dir=config['defect_segmentation']['save_dir']
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
                save_dir=config['fruit_detection']['save_dir']
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
                save_dir=config['defect_classification']['save_dir']
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
                save_dir=config['defect_segmentation']['save_dir']
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


if __name__ == "__main__":
    main()
