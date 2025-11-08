"""
Segmentation Validation module for the Fruit Defect Detection System.

This module provides specialized validation functions for segmentation models,
including comparison between predicted and ground truth masks, calculation of
segmentation-specific metrics, and visualization capabilities.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml
from pathlib import Path
from typing import Tuple, List, Dict, Optional


class SegmentationValidator:
    """
    A comprehensive validation class for segmentation models in the fruit defect detection system.
    """
    
    def __init__(self, model_path: str, data_path: str, output_dir: str = "validation_output"):
        """
        Initialize the segmentation validator.
        
        Args:
            model_path (str): Path to the trained segmentation model
            data_path (str): Path to the dataset YAML file
            output_dir (str): Directory to save validation results
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        self.model = YOLO(model_path)
        
        # Load dataset configuration
        with open(data_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
    
    def load_ground_truth_mask(self, image_path: str, labels_path: str) -> np.ndarray:
        """
        Load ground truth segmentation mask from YOLO format labels.
        
        Args:
            image_path (str): Path to the image file
            labels_path (str): Path to the directory containing label files
            
        Returns:
            numpy array: Ground truth segmentation mask
        """
        # Get the corresponding label file
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        label_file = img_name + '.txt'
        label_path = os.path.join(labels_path, label_file)
        
        if not os.path.exists(label_path):
            # If no label file exists, return an empty mask
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        # Read the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Create an empty mask
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process each line in the label file
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse the line: class_id x1 y1 x2 y2 x3 y3 ...
            values = line.split()
            class_id = int(values[0])
            coords = [float(x) for x in values[1:]]
            
            # Convert YOLO format to mask - coords are normalized polygon points
            if len(coords) >= 6 and len(coords) % 2 == 0:  # At least 3 points (6 coordinates)
                # Convert normalized coordinates to pixel coordinates
                polygon = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    polygon.append([x, y])
                
                # Create mask from polygon
                polygon = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(gt_mask, [polygon], 1)
        
        return gt_mask
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Intersection over Union between predicted and ground truth masks.
        
        Args:
            pred_mask (np.ndarray): Predicted segmentation mask
            gt_mask (np.ndarray): Ground truth segmentation mask
            
        Returns:
            float: IoU score
        """
        # Ensure both masks are binary
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def calculate_dice_coefficient(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Dice coefficient between predicted and ground truth masks.
        
        Args:
            pred_mask (np.ndarray): Predicted segmentation mask
            gt_mask (np.ndarray): Ground truth segmentation mask
            
        Returns:
            float: Dice coefficient
        """
        # Ensure both masks are binary
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Calculate intersection and union for Dice
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        pred_sum = pred_binary.sum()
        gt_sum = gt_binary.sum()
        
        # Calculate Dice coefficient
        dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0
        
        return dice
    
    def calculate_pixel_accuracy(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate pixel accuracy between predicted and ground truth masks.
        
        Args:
            pred_mask (np.ndarray): Predicted segmentation mask
            gt_mask (np.ndarray): Ground truth segmentation mask
            
        Returns:
            float: Pixel accuracy
        """
        # Ensure both masks are binary
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Calculate pixel accuracy
        correct_pixels = np.sum(pred_binary == gt_binary)
        total_pixels = gt_binary.size
        
        accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
        
        return accuracy
    
    def calculate_mean_absolute_error(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate mean absolute error between predicted and ground truth masks.
        
        Args:
            pred_mask (np.ndarray): Predicted segmentation mask
            gt_mask (np.ndarray): Ground truth segmentation mask
            
        Returns:
            float: Mean absolute error
        """
        # Ensure both masks are binary
        pred_binary = (pred_mask > 0).astype(np.float32)
        gt_binary = (gt_mask > 0).astype(np.float32)
        
        # Calculate MAE
        mae = np.mean(np.abs(pred_binary - gt_binary))
        
        return mae
    
    def get_predicted_masks(self, image_path: str) -> List[np.ndarray]:
        """
        Get predicted segmentation masks for an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            List[np.ndarray]: List of predicted masks
        """
        # Run inference
        results = self.model(image_path)
        
        predicted_masks = []
        
        if results and len(results) > 0:
            result_item = results[0] if isinstance(results, list) else results
            
            if hasattr(result_item, 'masks') and result_item.masks is not None:
                if len(result_item.masks) > 0:
                    # Get all masks and convert to numpy arrays
                    for mask_tensor in result_item.masks.data:
                        pred_mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                        predicted_masks.append(pred_mask)
        
        return predicted_masks
    
    def validate_single_image(self, image_path: str, labels_path: str) -> Dict[str, float]:
        """
        Validate segmentation for a single image.
        
        Args:
            image_path (str): Path to the image file
            labels_path (str): Path to the directory containing label files
            
        Returns:
            Dict[str, float]: Dictionary containing various metrics
        """
        # Load ground truth mask
        gt_mask = self.load_ground_truth_mask(image_path, labels_path)
        
        # Get predicted masks
        pred_masks = self.get_predicted_masks(image_path)
        
        # Combine all predicted masks into a single mask
        if pred_masks:
            combined_pred_mask = np.zeros_like(gt_mask)
            for pred_mask in pred_masks:
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                combined_pred_mask = np.logical_or(combined_pred_mask, pred_mask).astype(np.uint8)
        else:
            combined_pred_mask = np.zeros_like(gt_mask)
        
        # Calculate metrics
        iou = self.calculate_iou(combined_pred_mask, gt_mask)
        dice = self.calculate_dice_coefficient(combined_pred_mask, gt_mask)
        pixel_acc = self.calculate_pixel_accuracy(combined_pred_mask, gt_mask)
        mae = self.calculate_mean_absolute_error(combined_pred_mask, gt_mask)
        
        return {
            'iou': iou,
            'dice_coefficient': dice,
            'pixel_accuracy': pixel_acc,
            'mean_absolute_error': mae
        }
    
    def validate_dataset(self, images_path: str, labels_path: str, max_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Validate segmentation on an entire dataset.
        
        Args:
            images_path (str): Path to the images directory
            labels_path (str): Path to the labels directory
            max_samples (int, optional): Maximum number of samples to validate
            
        Returns:
            Dict[str, float]: Dictionary containing average metrics across the dataset
        """
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        all_metrics = {
            'iou': [],
            'dice_coefficient': [],
            'pixel_accuracy': [],
            'mean_absolute_error': []
        }
        
        for img_file in image_files:
            img_path = os.path.join(images_path, img_file)
            
            # Validate single image
            metrics = self.validate_single_image(img_path, labels_path)
            
            # Store metrics
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        # Calculate average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[key] = np.mean(values) if values else 0.0
            avg_metrics[f'{key}_std'] = np.std(values) if values else 0.0
        
        return avg_metrics
    
    def visualize_results(self, image_path: str, labels_path: str, output_path: str, alpha: float = 0.5):
        """
        Create a visualization showing ground truth and predicted segmentation masks overlaid on the image.
        
        Args:
            image_path (str): Path to the original image
            labels_path (str): Path to the directory containing label files
            output_path (str): Path to save the visualization
            alpha (float): Transparency for mask overlays (0.0-1.0)
        """
        # Load the original image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        gt_mask = self.load_ground_truth_mask(image_path, labels_path)
        
        # Get predicted masks
        pred_masks = self.get_predicted_masks(image_path)
        
        # Combine all predicted masks into a single mask
        if pred_masks:
            combined_pred_mask = np.zeros_like(gt_mask)
            for pred_mask in pred_masks:
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                combined_pred_mask = np.logical_or(combined_pred_mask, pred_mask).astype(np.uint8)
        else:
            combined_pred_mask = np.zeros_like(gt_mask)
        
        # Calculate metrics for display
        iou = self.calculate_iou(combined_pred_mask, gt_mask)
        dice = self.calculate_dice_coefficient(combined_pred_mask, gt_mask)
        
        # Create the visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask overlay
        gt_overlay = img_rgb.copy()
        gt_overlay[gt_mask > 0] = [255, 0, 0]  # Red for ground truth
        axes[1].imshow(cv2.addWeighted(img_rgb, 1 - alpha, gt_overlay, alpha, 0))
        axes[1].set_title(f'Ground Truth Mask\n(Defects in Red)')
        axes[1].axis('off')
        
        # Predicted mask overlay
        pred_overlay = img_rgb.copy()
        pred_overlay[combined_pred_mask > 0] = [0, 255, 0]  # Green for prediction
        axes[2].imshow(cv2.addWeighted(img_rgb, 1 - alpha, pred_overlay, alpha, 0))
        axes[2].set_title(f'Predicted Mask\n(Defects in Green)\nIoU: {iou:.3f}, Dice: {dice:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Segmentation visualization saved to {output_path}")
    
    def run_comprehensive_validation(self, split: str = 'val', max_samples: Optional[int] = None) -> Dict:
        """
        Run comprehensive validation on a dataset split.
        
        Args:
            split (str): Dataset split to validate ('train', 'val', or 'test')
            max_samples (int, optional): Maximum number of samples to validate
            
        Returns:
            Dict: Complete validation results
        """
        # Get paths for the specified split
        base_path = os.path.dirname(self.data_path)
        
        # Construct image and label paths based on dataset config
        images_rel_path = self.dataset_config.get(split, f'{split}/images')
        labels_rel_path = images_rel_path.replace('images', 'labels')
        
        images_path = os.path.join(base_path, images_rel_path)
        labels_path = os.path.join(base_path, labels_rel_path)
        
        # Resolve paths in case they contain relative components
        images_path = os.path.abspath(images_path)
        labels_path = os.path.abspath(labels_path)
        
        # Validate paths exist
        if not os.path.exists(images_path):
            raise ValueError(f"Images path does not exist: {images_path}")
        if not os.path.exists(labels_path):
            raise ValueError(f"Labels path does not exist: {labels_path}")
        
        print(f"Running comprehensive validation on {split} split...")
        print(f"Images path: {images_path}")
        print(f"Labels path: {labels_path}")
        
        # Validate the dataset
        avg_metrics = self.validate_dataset(images_path, labels_path, max_samples)
        
        # Generate visualizations for a subset of images
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        viz_dir = os.path.join(self.output_dir, f"{split}_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        num_visualizations = min(5, len(image_files))  # Visualize up to 5 samples
        for i in range(num_visualizations):
            img_file = image_files[i]
            img_path = os.path.join(images_path, img_file)
            viz_path = os.path.join(viz_dir, f"viz_{img_file}")
            self.visualize_results(img_path, labels_path, viz_path)
        
        # Prepare results
        results = {
            'model_path': self.model_path,
            'dataset_split': split,
            'dataset_path': self.data_path,
            'average_metrics': avg_metrics,
            'num_samples_evaluated': len(image_files) if max_samples is None else min(max_samples, len(image_files)),
            'visualization_dir': viz_dir
        }
        
        # Save results to a file
        results_path = os.path.join(self.output_dir, f"validation_results_{split}.json")
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Validation completed. Results saved to {results_path}")
        print(f"Visualizations saved to {viz_dir}")
        
        return results


if __name__ == "__main__":
    # Example usage
    print("SegmentationValidator module loaded. Use SegmentationValidator class to run validation.")