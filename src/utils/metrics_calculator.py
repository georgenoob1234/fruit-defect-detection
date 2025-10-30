"""
Comprehensive metrics calculation module for fruit defect detection system.

This module provides functions to calculate and save detailed model evaluation metrics
including inference time, throughput, precision, recall, F1-score, accuracy, hardware usage,
model size, OOD detection metrics, and performance distribution reports.
"""

import os
import json
import csv
import time
import numpy as np
import psutil
import GPUtil
import torch
import cv2
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import statistics
from ultralytics import YOLO
import yaml
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class MetricsCalculator:
    """
    A comprehensive metrics calculator for model evaluation in fruit defect detection system.
    """
    
    def __init__(self, model_path, data_path, output_dir="metrics", training_time=None):
        """
        Initialize the metrics calculator.
        
        Args:
            model_path (str): Path to the trained model
            data_path (str): Path to the dataset YAML file
            output_dir (str): Directory to save metrics (default: "metrics")
            training_time (float, optional): Training time in seconds to include in metrics
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.dataset_config = self._load_dataset_config(data_path)
        
        # Create metrics directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "model_path": model_path,
                "model_size_mb": 0,
                "model_loading_time": 0
            },
            "inference_time": {
                "p50": 0,
                "p90": 0,
                "p99": 0,
                "avg": 0,
                "min": float('inf'),
                "max": 0,
                "target_met": False
            },
            "throughput": {
                "fps": 0,
                "frames_processed": 0
            },
            "classification_metrics": {
                "overall": {
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "accuracy": 0
                },
                "per_class": {}
            },
            "hardware_usage": {
                "cpu_usage_percent": 0,
                "memory_usage_mb": 0,
                "gpu_usage_percent": 0,
                "gpu_memory_mb": 0,
                "gpu_available": False
            },
            "ood_detection": {
                "ood_detected_count": 0,
                "ood_detection_rate": 0,
                "ood_flags": []
            },
            "calibration_metrics": {},
            "stress_test_results": {},
            "validation_sets": {
                "train": {},
                "val": {},
                "test": {}
            }
        }
        
        # Add training time to metrics if provided
        if training_time is not None:
            self.metrics["training_time_seconds"] = training_time
        else:
            self.metrics["training_time_seconds"] = 0  # Default value for backward compatibility
        
        # Load model and calculate model size and loading time
        self.model, loading_time = self._load_model_with_timing()
        self.metrics["model_info"]["model_loading_time"] = loading_time
        self.metrics["model_info"]["model_size_mb"] = self._get_model_size_mb()
        
        # Check if GPU is available
        self.metrics["hardware_usage"]["gpu_available"] = torch.cuda.is_available()
        
    def _load_dataset_config(self, data_path):
        """Load dataset configuration from YAML file."""
        with open(data_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model_with_timing(self):
        """Load the model and measure loading time."""
        start_time = time.time()
        model = YOLO(self.model_path)
        loading_time = time.time() - start_time
        return model, loading_time
    
    def _get_model_size_mb(self):
        """Get the model file size in MB."""
        if os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)  # Convert to MB
        return 0
    
    def _collect_hardware_metrics(self):
        """Collect hardware usage metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / (1024 * 1024)
        
        # GPU usage (if available)
        gpu_percent = 0
        gpu_memory_mb = 0
        
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]  # Get first GPU
            gpu_percent = gpu.load * 100
            gpu_memory_mb = gpu.memoryUsed
        
        return {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_mb": memory_mb,
            "gpu_usage_percent": gpu_percent,
            "gpu_memory_mb": gpu_memory_mb
        }
    
    def calculate_inference_time_metrics(self, test_images_path, num_samples=100):
        """
        Calculate inference time metrics with p50/p90/p99 percentiles.
        
        Args:
            test_images_path (str): Path to test images directory
            num_samples (int): Number of samples to test (default: 100)
        """
        # Collect inference times
        inference_times = []
        image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit to num_samples
        image_files = image_files[:num_samples]
        
        for img_file in image_files:
            img_path = os.path.join(test_images_path, img_file)
            img = cv2.imread(img_path)
            
            start_time = time.time()
            # Run inference
            results = self.model(img)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)
        
        if inference_times:
            # Calculate percentiles
            self.metrics["inference_time"]["p50"] = np.percentile(inference_times, 50)
            self.metrics["inference_time"]["p90"] = np.percentile(inference_times, 90)
            self.metrics["inference_time"]["p99"] = np.percentile(inference_times, 99)
            self.metrics["inference_time"]["avg"] = np.mean(inference_times)
            self.metrics["inference_time"]["min"] = min(inference_times)
            self.metrics["inference_time"]["max"] = max(inference_times)
            
            # Check if target is met (≤200ms per frame)
            self.metrics["inference_time"]["target_met"] = self.metrics["inference_time"]["p90"] <= 200
    
    def calculate_throughput(self, test_images_path, duration_seconds=10):
        """
        Calculate throughput in frames per second (FPS).
        
        Args:
            test_images_path (str): Path to test images directory
            duration_seconds (int): Duration to measure FPS (default: 10)
        """
        image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            return
        
        start_time = time.time()
        frames_processed = 0
        
        while (time.time() - start_time) < duration_seconds:
            img_file = image_files[frames_processed % len(image_files)]
            img_path = os.path.join(test_images_path, img_file)
            img = cv2.imread(img_path)
            
            # Run inference
            results = self.model(img)
            
            frames_processed += 1
        
        elapsed_time = time.time() - start_time
        fps = frames_processed / elapsed_time
        
        self.metrics["throughput"]["fps"] = fps
        self.metrics["throughput"]["frames_processed"] = frames_processed
    def calculate_classification_metrics(self, test_data_path, task_type="detection", model_config_path="config/model_config.yaml"):
        """
        Calculate precision, recall, F1-score, and accuracy for overall and per-class.
        
        Args:
            test_data_path (str): Path to test dataset
            task_type (str): Type of task ("detection", "classification", or "segmentation")
            model_config_path (str): Path to model configuration file to validate classes
        """
        # Load model configuration to get actual classes
        model_classes = self._get_classes_from_model_config(model_config_path, task_type)
        
        # Load the dataset YAML file to get the actual paths and dataset classes
        with open(self.data_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Get dataset class names from the dataset config
        dataset_classes = dataset_config.get('names', [])
        if not dataset_classes:
            # Fallback to default classes if not found in dataset config
            dataset_classes = ["apple", "banana", "tomato"] if task_type == "detection" else ["defective", "non_defective"]
        
        # Create class mapping to handle ID mismatches between model and dataset
        class_mapping = self._create_class_mapping(dataset_classes, model_classes)
        
        # Get the test images and labels paths from the YAML config
        base_path = os.path.dirname(self.data_path)
        
        # Handle relative paths in dataset YAML (like '../test/images' in fruits_320/data.yaml)
        test_images_rel_path = dataset_config.get('test', 'test/images')
        test_labels_rel_path = dataset_config.get('test', 'test/labels').replace('images', 'labels')  # Convert images path to labels path
        
        # Resolve the paths relative to the dataset YAML file location
        test_images_path = os.path.join(base_path, test_images_rel_path)
        test_labels_path = os.path.join(base_path, test_labels_rel_path)
        
        # If the path starts with '../' or './', resolve it properly
        test_images_path = os.path.abspath(test_images_path)
        test_labels_path = os.path.abspath(test_labels_path)
        
        # If the resolved path doesn't exist, try alternative paths
        if not os.path.exists(test_images_path):
            # Try alternative paths based on common structures
            alt_paths = [
                os.path.join(base_path, 'test', 'images'),
                os.path.join(base_path, 'images', 'test'),
                os.path.join(base_path, test_images_rel_path),  # Try with original relative path
                test_data_path
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    test_images_path = alt_path
                    break
        
        if not os.path.exists(test_labels_path):
            # Try alternative paths for labels
            alt_paths = [
                os.path.join(base_path, 'test', 'labels'),
                os.path.join(base_path, 'labels', 'test'),
                os.path.join(base_path, test_labels_rel_path),  # Try with original relative path
                test_data_path.replace('images', 'labels') if 'images' in test_data_path else test_data_path
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    test_labels_path = alt_path
                    break
        
        # Calculate metrics based on actual predictions vs ground truth
        if task_type == "detection":
            # Calculate metrics for fruit detection using model classes and class mapping
            # Use the test_data_path parameter as fallback if test_images_path doesn't exist
            if not os.path.exists(test_images_path) and os.path.exists(test_data_path):
                test_images_path = test_data_path
            precision, recall, f1_score, accuracy, per_class_metrics = self._compute_detection_metrics(
                test_images_path, test_labels_path, model_classes, class_mapping, dataset_classes
            )
            
            self.metrics["classification_metrics"]["overall"]["precision"] = precision
            self.metrics["classification_metrics"]["overall"]["recall"] = recall
            self.metrics["classification_metrics"]["overall"]["f1_score"] = f1_score
            self.metrics["classification_metrics"]["overall"]["accuracy"] = accuracy
            
            # Per-class metrics for model classes
            self.metrics["classification_metrics"]["per_class"] = per_class_metrics
        
        elif task_type == "classification":
            # Calculate metrics for defect classification (defective/non_defective)
            precision, recall, f1_score, accuracy, per_class_metrics = self._compute_classification_metrics(
                test_images_path, test_labels_path, model_classes
            )
            
            self.metrics["classification_metrics"]["overall"]["precision"] = precision
            self.metrics["classification_metrics"]["overall"]["recall"] = recall
            self.metrics["classification_metrics"]["overall"]["f1_score"] = f1_score
            self.metrics["classification_metrics"]["overall"]["accuracy"] = accuracy
            
            # Per-class metrics for model classes
            self.metrics["classification_metrics"]["per_class"] = per_class_metrics

    def _compute_detection_metrics(self, images_path, labels_path, class_names, class_mapping=None, dataset_class_names=None):
        """
        Compute detection metrics by comparing model predictions to ground truth annotations.
        Handles class ID mapping between dataset and model to account for different class orderings.
        
        Args:
            images_path (str): Path to test images
            labels_path (str): Path to ground truth labels
            class_names (list): List of model class names
            class_mapping (dict, optional): Mapping from dataset class ID to model class ID
            dataset_class_names (list, optional): List of dataset class names
            
        Returns:
            tuple: (precision, recall, f1_score, accuracy, per_class_metrics)
        """
        # Initialize counters for overall metrics
        tp_total = 0  # True positives
        fp_total = 0  # False positives
        fn_total = 0  # False negatives
        
        # Initialize per-class counters using model class names
        class_stats = {class_name: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0} for class_name in class_names}
        
        # Get all image files
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"DEBUG: Processing {len(image_files)} images for detection metrics")
        print(f"DEBUG: Model class names: {class_names}")
        print(f"DEBUG: Dataset class names: {dataset_class_names}")
        print(f"DEBUG: Class mapping: {class_mapping}")
        
        # Process each image - remove limit for full processing
        for img_idx, img_file in enumerate(image_files):
            # Get corresponding label file
            img_name = os.path.splitext(img_file)[0]
            label_file = img_name + '.txt'
            label_path = os.path.join(labels_path, label_file)
            
            if not os.path.exists(label_path):
                print(f"DEBUG: Label file {label_path} does not exist, skipping")
                continue # Skip if no ground truth label exists
            
            # Load ground truth with dataset class names and apply mapping if provided
            gt_boxes = self._load_ground_truth_labels(label_path, dataset_class_names)
            
            # Load image and run model prediction
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            
            # Run inference
            results = self.model(img)
            
            # Extract predictions
            pred_boxes = []
            print(f"DEBUG: Raw results type: {type(results)}")
            
            # Handle case where results is a list (common with YOLO models)
            if isinstance(results, list):
                print(f"DEBUG: Results is a list with {len(results)} items")
                # Take the first item if it's a list
                if len(results) > 0:
                    result_item = results[0]
                else:
                    result_item = None
            else:
                result_item = results
            
            # Now process the result item
            if result_item is not None and hasattr(result_item, 'boxes') and result_item.boxes is not None:
                print(f"DEBUG: Result item has boxes attribute, length: {len(result_item.boxes)}")
                if len(result_item.boxes) > 0:
                    boxes = result_item.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confs = result_item.boxes.conf.cpu().numpy()
                    cls_ids = result_item.boxes.cls.cpu().numpy()
                    
                    print(f"DEBUG: Raw predictions - boxes: {len(boxes)}, confs: {confs}, cls_ids: {cls_ids}")
                    print(f"DEBUG: Model class names: {class_names}")
                    
                    # Apply confidence threshold - lowered from 0.5 to 0.25 to catch more predictions
                    valid_indices = confs >= 0.25  # Lowered confidence threshold to capture more predictions
                    boxes = boxes[valid_indices]
                    cls_ids = cls_ids[valid_indices].astype(int)
                    confs = confs[valid_indices]
                    
                    print(f"DEBUG: After confidence filter - boxes: {len(boxes)}, cls_ids: {cls_ids}")
                    
                    # Get image dimensions for coordinate conversion
                    img_height, img_width = img.shape[0], img.shape[1]
                    
                    for i in range(len(boxes)):
                       # Add debug info to see what classes are being predicted
                       print(f"DEBUG: Processing prediction {i}, class_id: {cls_ids[i]}, confidence: {confs[i]}")
                       if cls_ids[i] < len(class_names):
                           pred_class_name = class_names[cls_ids[i]]
                           # Convert pixel coordinates to normalized coordinates to match ground truth
                           x1, y1, x2, y2 = boxes[i]
                           center_x = ((x1 + x2) / 2) / img_width  # Normalize x
                           center_y = ((y1 + y2) / 2) / img_height  # Normalize y
                           width = (x2 - x1) / img_width  # Normalize width
                           height = (y2 - y1) / img_height  # Normalize height
                           
                           pred_boxes.append({
                               'class_id': cls_ids[i],
                               'class_name': pred_class_name,
                               'bbox': [center_x, center_y, width, height],  # normalized coordinates to match ground truth
                               'confidence': confs[i]
                           })
                           print(f"DEBUG: Added prediction {i} - class: {pred_class_name}, confidence: {confs[i]:.3f}")
                       else:
                           # Handle case where model predicts a class ID not in our class_names list
                           print(f"DEBUG: Model predicted class_id {cls_ids[i]} which is not in class_names {class_names}")
                           # Use a default name or skip
                           continue
                else:
                    print(f"DEBUG: No boxes in result item")
            else:
                print(f"DEBUG: Result item is None or does not have boxes attribute or boxes is None")
                # Try to see what attributes result_item has
                if result_item is not None:
                    print(f"DEBUG: Result item attributes: {[attr for attr in dir(result_item) if not attr.startswith('_')]}")
                else:
                    print(f"DEBUG: Result item is None")
            
            print(f"DEBUG: Image {img_idx}: {len(gt_boxes)} GT boxes, {len(pred_boxes)} pred boxes")
            
            # Update class statistics with ground truth counts
            for gt_box in gt_boxes:
                class_name = gt_box['class_name']
                if class_name in class_stats:
                    class_stats[class_name]["total_gt"] += 1
            
            # Perform IoU-based matching between ground truth and predictions
            matched_pred_indices = set()  # Track which predictions have been matched
            
            # For each ground truth box, find the best matching prediction
            for gt_idx, gt_box in enumerate(gt_boxes):
                best_iou = 0
                best_match_idx = -1
                best_pred_box = None
                
                # Find the best matching prediction for this ground truth box
                for pred_idx, pred_box in enumerate(pred_boxes):
                    if pred_idx in matched_pred_indices:
                        continue # Skip already matched predictions
                    
                    # Check if classes match
                    if gt_box['class_id'] != pred_box['class_id']:
                        continue  # Skip if classes don't match
                    
                    # Calculate IoU between gt_box and pred_box
                    iou = self._calculate_iou(gt_box['bbox'], pred_box['bbox'])
                    
                    if iou > best_iou and iou >= 0.5:  # Standard IoU threshold
                        best_iou = iou
                        best_match_idx = pred_idx
                        best_pred_box = pred_box
                
                # Handle the match
                class_name = gt_box['class_name']
                if class_name in class_stats:
                    if best_match_idx != -1:
                        # Found a match (true positive)
                        matched_pred_indices.add(best_match_idx)
                        class_stats[class_name]["tp"] += 1
                        tp_total += 1
                        print(f"DEBUG: Matched GT {gt_idx} ({class_name}) with pred {best_match_idx} ({best_pred_box['class_name']}) IoU={best_iou:.2f}")
                    else:
                        # No match found for this ground truth (false negative)
                        class_stats[class_name]["fn"] += 1
                        fn_total += 1
                        print(f"DEBUG: Unmatched GT {gt_idx} ({class_name})")
            
            # Handle unmatched predictions (false positives)
            for pred_idx, pred_box in enumerate(pred_boxes):
                if pred_idx not in matched_pred_indices:
                    class_name = pred_box['class_name']
                    if class_name in class_stats:
                        class_stats[class_name]["fp"] += 1
                        fp_total += 1
                        print(f"DEBUG: Unmatched pred {pred_idx} ({class_name})")
        
        print(f"DEBUG: Class statistics:")
        for class_name, stats in class_stats.items():
            print(f"  {class_name}: TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}, Total GT={stats['total_gt']}")
        
        # Calculate overall metrics
        if (tp_total + fp_total) > 0:
            precision = tp_total / (tp_total + fp_total)
        else:
            precision = 0.0
            
        if (tp_total + fn_total) > 0:
            recall = tp_total / (tp_total + fn_total)
        else:
            recall = 0.0
            
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
            
        # For accuracy in object detection, we'll use a simple approximation
        total_gt = sum(stats["total_gt"] for stats in class_stats.values())
        if total_gt > 0:
            accuracy = tp_total / total_gt
        else:
            accuracy = 0.0
        
        print(f"DEBUG: Overall metrics - P: {precision:.3f}, R: {recall:.3f}, F1: {f1_score:.3f}, Acc: {accuracy:.3f}")
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for class_name in class_names:
            stats = class_stats[class_name]
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            total_gt = stats["total_gt"]
            
            # Calculate per-class metrics
            if (tp + fp) > 0:
                class_precision = tp / (tp + fp)
            else:
                class_precision = 0.0
                
            if (tp + fn) > 0:
                class_recall = tp / (tp + fn)
            else:
                class_recall = 0.0
                
            if (class_precision + class_recall) > 0:
                class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
            else:
                class_f1 = 0.0
                
            if total_gt > 0:
                class_accuracy = tp / total_gt
            else:
                class_accuracy = 0.0
            
            per_class_metrics[class_name] = {
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1_score": float(class_f1),
                "accuracy": float(class_accuracy)
            }
            
            print(f"DEBUG: Class {class_name} metrics - P: {class_precision:.3f}, R: {class_recall:.3f}, F1: {class_f1:.3f}, Acc: {class_accuracy:.3f}")
        
        return float(precision), float(recall), float(f1_score), float(accuracy), per_class_metrics

    def _load_ground_truth_labels(self, label_path, class_names, class_mapping=None):
        """
        Load ground truth labels from YOLO format text file.
        Handles class ID mapping between dataset and model to account for different class orderings.
        
        Args:
            label_path (str): Path to the label file
            class_names (list): List of class names
            class_mapping (dict, optional): Mapping from dataset class ID to model class ID
            
        Returns:
            list: List of dictionaries with class_id, class_name, and bbox
        """
        boxes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # YOLO format: class_id center_x center_y width height (normalized)
                    parts = line.split()
                    if len(parts) >= 5:
                        dataset_class_id = int(parts[0])
                        if dataset_class_id < len(class_names):
                            # Map dataset class ID to model class ID if mapping provided
                            model_class_id = class_mapping.get(dataset_class_id, dataset_class_id) if class_mapping else dataset_class_id
                            
                            # Parse the bounding box values
                            center_x, center_y, width, height = [float(x) for x in parts[1:5]]
                            
                            # Convert normalized coordinates to pixel coordinates
                            # This is handled during IoU calculation
                            boxes.append({
                                'class_id': model_class_id,
                                'class_name': class_names[model_class_id] if model_class_id < len(class_names) else class_names[dataset_class_id],
                                'bbox': [center_x, center_y, width, height]  # normalized coordinates
                            })
        except Exception as e:
            print(f"Error loading ground truth labels from {label_path}: {e}")
        
        return boxes

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes.
        This handles normalized coordinates (0-1 range) from YOLO format.
        
        Args:
            box1, box2: Bounding boxes in format [center_x, center_y, width, height] (normalized)
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Convert from [center_x, center_y, width, height] to [x1, y1, x2, y2]
        def convert_to_xyxy(bbox):
            center_x, center_y, width, height = bbox
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            return x1, y1, x2, y2
        
        x1_1, y1_1, x2_1, y2_1 = convert_to_xyxy(box1)
        x1_2, y1_2, x2_2, y2_2 = convert_to_xyxy(box2)
        
        # Calculate intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        # Calculate intersection area
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        
        # Calculate areas of both boxes
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        union_area = area_1 + area_2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou

    def _get_classes_from_model_config(self, model_config_path, task_type):
        """
        Get actual classes from model configuration file or from the loaded model itself.
        
        Args:
            model_config_path (str): Path to model configuration file
            task_type (str): Type of task ("detection", "classification", or "segmentation")
        
        Returns:
            list: List of actual class names from the configuration or model
        """
        # First, try to get classes from the loaded model directly
        try:
            # YOLO model has a names attribute that contains the class names
            if hasattr(self.model, 'names') and self.model.names is not None:
                model_names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
                print(f"DEBUG: Model class names: {model_names}")
                return model_names
        except Exception as e:
            print(f"Could not get classes from model: {e}")
        
        # Fallback to loading from config file
        if os.path.exists(model_config_path):
            try:
                with open(model_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Determine which section to use based on task type
                if task_type == "detection":
                    section = "fruit_detection"
                elif task_type == "classification":
                    section = "defect_detection"  # Using the defect detection section for classification
                elif task_type == "segmentation":
                    section = "defect_detection"  # Using the defect detection section for segmentation
                else:
                    section = "fruit_detection"  # Default fallback
                
                if section in config and "target_classes" in config[section]:
                    return config[section]["target_classes"]
                elif section in config and "names" in config[section]:
                    return config[section]["names"]
                else:
                    # Default classes if not found in config
                    if task_type == "detection":
                        return ["apple", "banana", "tomato"]
                    elif task_type in ["classification", "segmentation"]:
                        return ["defective", "non_defective"]
            except Exception as e:
                print(f"Error loading model configuration: {e}")
                # Return default classes if config loading fails
                if task_type == "detection":
                    return ["apple", "banana", "tomato"]
                elif task_type in ["classification", "segmentation"]:
                    return ["defective", "non_defective"]
        else:
            # Return default classes if config file doesn't exist
            if task_type == "detection":
                return ["apple", "banana", "tomato"]
            elif task_type in ["classification", "segmentation"]:
                return ["defective", "non_defective"]
        
        return []
    
    def _create_class_mapping(self, dataset_class_names, model_class_names):
        """
        Create mapping between dataset class names and model class names to handle ID mismatches.
        
        Args:
            dataset_class_names (list): Class names from dataset configuration
            model_class_names (list): Class names from model
            
        Returns:
            dict: Mapping from dataset class ID to model class ID
        """
        # Create mapping from class names to IDs for both dataset and model
        dataset_name_to_id = {name: idx for idx, name in enumerate(dataset_class_names)}
        model_name_to_id = {name: idx for idx, name in enumerate(model_class_names)}
        
        # Create mapping from dataset class ID to model class ID
        class_mapping = {}
        for class_name in dataset_class_names:
            if class_name in dataset_name_to_id and class_name in model_name_to_id:
                dataset_id = dataset_name_to_id[class_name]
                model_id = model_name_to_id[class_name]
                class_mapping[dataset_id] = model_id
            else:
                # If class name doesn't exist in one of them, use identity mapping
                if class_name in dataset_name_to_id:
                    dataset_id = dataset_name_to_id[class_name]
                    class_mapping[dataset_id] = dataset_id
        
        print(f"DEBUG: Class mapping (dataset_id -> model_id): {class_mapping}")
        return class_mapping

    
    def calculate_ood_detection_metrics(self, ood_images_path):
        """
        Calculate Out-of-Distribution detection metrics.
        
        Args:
            ood_images_path (str): Path to out-of-distribution images
        """
        # This is a simplified implementation - in practice you'd need a method
        # to identify OOD samples based on model confidence or other metrics
        
        ood_image_files = [f for f in os.listdir(ood_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # For demonstration, we'll simulate OOD detection based on low confidence
        ood_count = 0
        
        for img_file in ood_image_files[:50]:  # Limit to first 50 for performance
            img_path = os.path.join(ood_images_path, img_file)
            img = cv2.imread(img_path)
            
            results = self.model(img)
            
            # Check if confidence is low (indicating potential OOD)
            if hasattr(results, 'boxes') and results.boxes is not None:
                if len(results.boxes) > 0:
                    confidences = results.boxes.conf.cpu().numpy()
                    if len(confidences) > 0 and np.mean(confidences) < 0.3:  # Low confidence
                        ood_count += 1
                else:
                    ood_count += 1  # No detections might indicate OOD
            else:
                ood_count += 1  # No results might indicate OOD
        
        total_samples = min(50, len(ood_image_files))
        self.metrics["ood_detection"]["ood_detected_count"] = ood_count
        self.metrics["ood_detection"]["ood_detection_rate"] = ood_count / total_samples if total_samples > 0 else 0
    
    def calculate_performance_distribution(self, test_images_path, num_samples=200):
        """
        Generate performance distribution reports showing frame processing times.
        
        Args:
            test_images_path (str): Path to test images directory
            num_samples (int): Number of samples to test (default: 200)
        """
        # Collect processing times for distribution analysis
        processing_times = []
        image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit to num_samples
        image_files = image_files[:num_samples]
        
        for img_file in image_files:
            img_path = os.path.join(test_images_path, img_file)
            img = cv2.imread(img_path)
            
            start_time = time.time()
            results = self.model(img)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
            processing_times.append(processing_time)
        
        # Calculate distribution metrics
        if processing_times:
            self.metrics["performance_distribution"] = {
                "mean_processing_time": statistics.mean(processing_times),
                "median_processing_time": statistics.median(processing_times),
                "std_dev_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "processing_times_samples": processing_times[:50]  # Store first 50 samples for analysis
            }
    
    def run_calibration_metrics(self, calibration_images_path):
        """
        Run calibration metrics for the model.
        
        Args:
            calibration_images_path (str): Path to calibration images
        """
        # Collect confidence scores for calibration analysis
        all_confidences = []
        
        image_files = [f for f in os.listdir(calibration_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files[:100]:  # Limit to first 100 for performance
            img_path = os.path.join(calibration_images_path, img_file)
            img = cv2.imread(img_path)
            
            results = self.model(img)
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                confidences = results.boxes.conf.cpu().numpy()
                all_confidences.extend(confidences)
        
        if all_confidences:
            # Calculate calibration metrics
            self.metrics["calibration_metrics"] = {
                "mean_confidence": np.mean(all_confidences),
                "confidence_std": np.std(all_confidences),
                "confidence_range": (np.min(all_confidences), np.max(all_confidences)),
                "confidence_histogram": np.histogram(all_confidences, bins=20)[0].tolist()
            }
        # Debug output to understand why calibration metrics might be empty
        print(f"DEBUG: Calibration images path: {calibration_images_path}")
        if calibration_images_path and os.path.exists(calibration_images_path):
            print(f"DEBUG: Calibration path exists, image count: {len(os.listdir(calibration_images_path))}")
        else:
            print(f"DEBUG: Calibration path does not exist or was not provided")
    
    def run_stress_testing(self, stress_test_images_path):
        """
        Run stress testing for poor quality images (glare/shadows/blur).
        
        Args:
            stress_test_images_path (str): Path to stress test images
        """
        # This would involve testing with images that have glare, shadows, blur, etc.
        # For this implementation, we'll categorize images by their quality metrics
        
        stress_results = {
            "glare_images": {"count": 0, "success_rate": 0},
            "shadow_images": {"count": 0, "success_rate": 0},
            "blur_images": {"count": 0, "success_rate": 0},
            "low_light_images": {"count": 0, "success_rate": 0}
        }
        
        image_files = [f for f in os.listdir(stress_test_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files[:100]:  # Limit to first 100 for performance
            img_path = os.path.join(stress_test_images_path, img_file)
            img = cv2.imread(img_path)
            
            # Calculate image quality metrics
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect glare (very bright areas)
            glare_score = np.mean(gray[gray > 200]) if np.any(gray > 200) else 0
            has_glare = glare_score > 150
            
            # Detect blur (low variance in Laplacian)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blur = laplacian_var < 100  # Threshold for blur detection
            
            # Detect shadows (dark areas)
            shadow_score = np.mean(gray[gray < 50]) if np.any(gray < 50) else 0
            has_shadow = shadow_score < 30
            
            # Detect low light
            avg_brightness = np.mean(gray)
            is_low_light = avg_brightness < 80
            
            # Run inference
            results = self.model(img)
            success = len(results.boxes) > 0 if hasattr(results, 'boxes') and results.boxes is not None else False
            
            # Categorize and record results
            if has_glare:
                stress_results["glare_images"]["count"] += 1
                if success:
                    stress_results["glare_images"]["success_rate"] += 1
            if has_shadow:
                stress_results["shadow_images"]["count"] += 1
                if success:
                    stress_results["shadow_images"]["success_rate"] += 1
            if is_blur:
                stress_results["blur_images"]["count"] += 1
                if success:
                    stress_results["blur_images"]["success_rate"] += 1
            if is_low_light:
                stress_results["low_light_images"]["count"] += 1
                if success:
                    stress_results["low_light_images"]["success_rate"] += 1
        
        # Calculate success rates
        for category in stress_results:
            if stress_results[category]["count"] > 0:
                stress_results[category]["success_rate"] /= stress_results[category]["count"]
        
        # Debug output to understand why stress test results might be empty
        print(f"DEBUG: Stress test images path: {stress_test_images_path}")
        if stress_test_images_path and os.path.exists(stress_test_images_path):
            print(f"DEBUG: Stress test path exists, image count: {len(os.listdir(stress_test_images_path))}")
        else:
            print(f"DEBUG: Stress test path does not exist or was not provided")
        self.metrics["stress_test_results"] = stress_results
    
    def run_validation_on_sets(self, data_path):
        """
        Run validation on train/val/test sets with emphasis on test set performance.
        
        Args:
            data_path (str): Path to dataset YAML configuration
        """
        # Load dataset configuration
        with open(data_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Get the directory of the data_path to use as base path if no 'path' key exists
        base_path = os.path.dirname(data_path)
        
        # Debug output to understand why validation sets might be empty
        print(f"DEBUG: Dataset config path: {data_path}")
        print(f"DEBUG: Dataset config content: {dataset_config}")
        for set_name in ["train", "val", "test"]:
            if set_name in dataset_config:
                if "path" in dataset_config:
                    set_path = os.path.join(dataset_config["path"], dataset_config[set_name])
                else:
                    # The dataset path is relative to the YAML file location
                    # Need to resolve relative paths properly (e.g., ../train/images from data/apples_320_null/data.yaml)
                    set_path = os.path.join(base_path, dataset_config[set_name])
                    # Resolve the path to handle relative path components like ../
                    set_path = os.path.normpath(set_path)
                    
                    # If the resolved path doesn't exist, try looking in the same directory as the YAML file
                    if not os.path.exists(set_path):
                        # Try the path directly in the same directory as the YAML file
                        alt_path = os.path.join(base_path, os.path.basename(dataset_config[set_name]))
                        if os.path.exists(alt_path):
                            set_path = alt_path
                        else:
                            # If that doesn't work, try the path without the first directory component
                            path_parts = dataset_config[set_name].split('/')
                            if len(path_parts) >= 2:
                                alt_path = os.path.join(base_path, *path_parts[1:])
                                if os.path.exists(alt_path):
                                    set_path = alt_path
                                    
                    print(f"DEBUG: {set_name} set path: {set_path}, exists: {os.path.exists(set_path)}")
                if os.path.exists(set_path):
                    image_files = [f for f in os.listdir(set_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    print(f"DEBUG: {set_name} set image count: {len(image_files)}")
        
        # Validate on each set (train, val, test)
        for set_name in ["train", "val", "test"]:
            if set_name in dataset_config:
                # If dataset_config has a 'path' key, use it; otherwise use the directory of the data file
                if "path" in dataset_config:
                    set_path = os.path.join(dataset_config["path"], dataset_config[set_name])
                else:
                    # The dataset path is relative to the YAML file location
                    # We need to properly resolve relative paths like ../train/images
                    set_path = os.path.join(base_path, dataset_config[set_name])
                    # Resolve the path to handle relative path components like ../
                    set_path = os.path.normpath(set_path)
                    
                    # If the resolved path doesn't exist, try looking in the same directory as the YAML file
                    # This handles cases where the YAML file has relative paths like ../train/images but
                    # the actual directories are in the same directory as the YAML file
                    if not os.path.exists(set_path):
                        # Try the path directly in the same directory as the YAML file
                        alt_path = os.path.join(base_path, os.path.basename(dataset_config[set_name]))
                        if os.path.exists(alt_path):
                            set_path = alt_path
                        else:
                            # If that doesn't work, try the path without the first directory component
                            # For example, if the YAML has '../train/images' and we're in 'data/apples_320_null',
                            # the resolved path would be 'data/train/images' which doesn't exist,
                            # but 'data/apples_320_null/train/images' might exist
                            path_parts = dataset_config[set_name].split('/')
                            if len(path_parts) >= 2:
                                alt_path = os.path.join(base_path, *path_parts[1:])  # Skip the first part (e.g., '..')
                                if os.path.exists(alt_path):
                                    set_path = alt_path
                
                # Count images in the set
                if os.path.exists(set_path):
                    image_files = [f for f in os.listdir(set_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    # For test set, ensure minimum requirements
                    if set_name == "test":
                        # Check if we have minimum 120 images, ≥40 per class, ≥30 problematic examples
                        min_images_met = len(image_files) >= 120
                        # This would require more detailed analysis of class distribution
                        
                        self.metrics["validation_sets"][set_name] = {
                            "image_count": len(image_files),
                            "min_images_requirement_met": min_images_met,
                            "processing_time": self._estimate_processing_time(set_path, image_files[:50])
                        }
                    else:
                        self.metrics["validation_sets"][set_name] = {
                            "image_count": len(image_files),
                            "processing_time": self._estimate_processing_time(set_path, image_files[:20])
                        }
    
    def _estimate_processing_time(self, set_path, image_files):
        """Estimate average processing time for a set."""
        if not image_files:
            return 0
        
        total_time = 0
        for img_file in image_files:
            img_path = os.path.join(set_path, img_file)
            img = cv2.imread(img_path)
            
            start_time = time.time()
            self.model(img)
            end_time = time.time()
            
            total_time += (end_time - start_time)
        
        return total_time / len(image_files) if image_files else 0
    
    def validate_kpi_targets(self):
        """Validate metrics against KPI targets."""
        # Check KPI targets
        kpi_results = {
            "recall_for_problematic_cases_met": self.metrics["classification_metrics"]["overall"]["recall"] >= 0.80,
            "precision_for_marked_items_met": self.metrics["classification_metrics"]["overall"]["precision"] >= 0.85,
            "unnecessary_suggestions_met": (1 - self.metrics["classification_metrics"]["overall"]["precision"]) <= 0.10
        }
        
        self.metrics["kpi_validation"] = kpi_results
    
    def calculate_all_metrics(self, test_images_path, ood_images_path=None, calibration_images_path=None, stress_test_images_path=None, model_config_path="config/model_config.yaml"):
        """
        Calculate all metrics in one comprehensive function.
        
        Args:
            test_images_path (str): Path to test images
            ood_images_path (str, optional): Path to out-of-distribution images
            calibration_images_path (str, optional): Path to calibration images
            stress_test_images_path (str, optional): Path to stress test images
            model_config_path (str): Path to model configuration file to validate classes
        """
        print("Starting comprehensive metrics calculation...")
        
        # Collect hardware metrics
        hardware_metrics = self._collect_hardware_metrics()
        self.metrics["hardware_usage"].update(hardware_metrics)
        
        # Calculate inference time metrics
        print("Calculating inference time metrics...")
        self.calculate_inference_time_metrics(test_images_path)
        
        # Calculate throughput
        print("Calculating throughput metrics...")
        self.calculate_throughput(test_images_path)
        
        # Calculate classification metrics (assuming detection task)
        print("Calculating classification metrics...")
        self.calculate_classification_metrics(test_images_path, task_type="detection", model_config_path=model_config_path)
        
        # Calculate OOD detection metrics if OOD images path provided
        if ood_images_path and os.path.exists(ood_images_path):
            print("Calculating OOD detection metrics...")
            self.calculate_ood_detection_metrics(ood_images_path)
        
        # Calculate performance distribution
        print("Calculating performance distribution...")
        self.calculate_performance_distribution(test_images_path)
        
        # Run calibration metrics if calibration images path provided
        if calibration_images_path and os.path.exists(calibration_images_path):
            print("Running calibration metrics...")
            self.run_calibration_metrics(calibration_images_path)
        
        # Run stress testing if stress test images path provided
        if stress_test_images_path and os.path.exists(stress_test_images_path):
            print("Running stress testing...")
            self.run_stress_testing(stress_test_images_path)
        
        # Run validation on sets
        print("Running validation on sets...")
        self.run_validation_on_sets(self.data_path)
        
        # Validate KPI targets
        print("Validating KPI targets...")
        self.validate_kpi_targets()
        
        print("Comprehensive metrics calculation completed!")
    
    def save_metrics(self, format="json"):
        """
        Save metrics to file in specified format.
        
        Args:
            format (str): Output format ("json" or "csv")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}"
        
        if format.lower() == "json":
            filepath = os.path.join(self.output_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
        elif format.lower() == "csv":
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
            self._save_metrics_csv(filepath)
        
        print(f"Metrics saved to {filepath}")
        return filepath
    
    def _save_metrics_csv(self, filepath):
        """Save metrics to CSV format."""
        # Flatten the metrics dictionary for CSV
        flattened_metrics = self._flatten_dict(self.metrics)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['metric', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for key, value in flattened_metrics.items():
                writer.writerow({'metric': key, 'value': str(value)})
    
    def _flatten_dict(self, d, parent_key='', sep='.'):
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

def calculate_comprehensive_metrics(model_path, data_path, test_images_path,
                                  ood_images_path=None, calibration_images_path=None,
                                  stress_test_images_path=None, output_dir="metrics",
                                  model_config_path="config/model_config.yaml", training_time=None):

    """
    Main function to calculate comprehensive metrics for a trained model.
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to the dataset YAML file
        test_images_path (str): Path to test images
        ood_images_path (str, optional): Path to out-of-distribution images
        calibration_images_path (str, optional): Path to calibration images
        stress_test_images_path (str, optional): Path to stress test images
        output_dir (str): Directory to save metrics (default: "metrics")
        training_time (float, optional): Training time in seconds to include in metrics
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Create metrics calculator instance
    calculator = MetricsCalculator(model_path, data_path, output_dir, training_time=training_time)
    
    # Calculate all metrics
    calculator.calculate_all_metrics(
        test_images_path=test_images_path,
        ood_images_path=ood_images_path,
        calibration_images_path=calibration_images_path,
        stress_test_images_path=stress_test_images_path,
        model_config_path=model_config_path
    )
    # Save metrics in both JSON and CSV formats
    json_path = calculator.save_metrics("json")
    csv_path = calculator.save_metrics("csv")
    
    print(f"Metrics saved to {json_path} and {csv_path}")
    
    return calculator.metrics


if __name__ == "__main__":
    # Example usage
    # This would be called from the training script after model training
    print("Metrics calculator module loaded. Use calculate_comprehensive_metrics() function to run evaluation.")