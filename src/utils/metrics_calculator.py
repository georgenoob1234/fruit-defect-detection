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
            "segmentation_metrics": {
                "overall": {
                    "iou": 0,
                    "dice_coefficient": 0,
                    "pixel_accuracy": 0,
                    "mean_absolute_error": 0
                },
                "per_class": {}
            },
            "object_detection_metrics": {
                "mAP": {
                    "overall": 0,
                    "per_iou": {
                        "0.5": 0,
                        "0.75": 0,
                        "0.9": 0
                    },
                    "per_class": {}
                },
                "AP": {
                    "overall": 0,
                    "per_class": {}
                },
                "AR": {
                    "overall": 0,
                    "per_class": {}
                },
                "true_positives": {
                    "per_class": {},
                    "overall": 0
                },
                "false_positives": {
                    "per_class": {},
                    "overall": 0
                },
                "false_negatives": {
                    "per_class": {},
                    "overall": 0
                },
                "mean_iou": {
                    "overall": 0,
                    "per_class": {}
                },
                "center_point_accuracy": {
                    "overall": 0,
                    "per_class": {}
                },
                "bbox_size_error": {
                    "overall": 0,
                    "per_class": {}
                },
                "false_positive_rate": {
                    "overall": 0,
                    "per_class": {}
                }
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
            
            # Calculate comprehensive object detection metrics
            self._compute_object_detection_metrics(
                test_images_path, test_labels_path, model_classes, class_mapping, dataset_classes
            )
            
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
        
        elif task_type == "segmentation":
            # Calculate metrics for segmentation tasks
            # Use test_labels_path for YOLO format labels and test_images_path for images
            iou_score, dice_score, pixel_accuracy, mae, per_class_seg_metrics = self._compute_segmentation_metrics(
                test_images_path, test_labels_path, model_classes
            )
            
            self.metrics["segmentation_metrics"]["overall"]["iou"] = iou_score
            self.metrics["segmentation_metrics"]["overall"]["dice_coefficient"] = dice_score
            self.metrics["segmentation_metrics"]["overall"]["pixel_accuracy"] = pixel_accuracy
            self.metrics["segmentation_metrics"]["overall"]["mean_absolute_error"] = mae
            
            # Per-class segmentation metrics
            self.metrics["segmentation_metrics"]["per_class"] = per_class_seg_metrics

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

    def _compute_object_detection_metrics(self, images_path, labels_path, class_names, class_mapping=None, dataset_class_names=None):
        """
        Compute comprehensive object detection metrics including mAP, AP, AR, and bounding box regression metrics.
        
        Args:
            images_path (str): Path to test images
            labels_path (str): Path to ground truth labels
            class_names (list): List of model class names
            class_mapping (dict, optional): Mapping from dataset class ID to model class ID
            dataset_class_names (list, optional): List of dataset class names
        """
        # Initialize data structures for storing detections and ground truths
        all_detections = {class_name: [] for class_name in class_names}
        all_ground_truths = {class_name: [] for class_name in class_names}
        
        # Get all image files
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"DEBUG: Computing object detection metrics for {len(image_files)} images")
        
        # Process each image
        for img_idx, img_file in enumerate(image_files):
            # Get corresponding label file
            img_name = os.path.splitext(img_file)[0]
            label_file = img_name + '.txt'
            label_path = os.path.join(labels_path, label_file)
            
            if not os.path.exists(label_path):
                continue  # Skip if no ground truth label exists
            
            # Load ground truth with dataset class names and apply mapping if provided
            gt_boxes = self._load_ground_truth_labels(label_path, dataset_class_names)
            
            # Load image and run model prediction
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            
            # Run inference
            results = self.model(img)
            
            # Extract predictions
            pred_boxes = []
            
            # Handle case where results is a list (common with YOLO models)
            if isinstance(results, list):
                result_item = results[0] if len(results) > 0 else None
            else:
                result_item = results
            
            # Now process the result item
            if result_item is not None and hasattr(result_item, 'boxes') and result_item.boxes is not None:
                if len(result_item.boxes) > 0:
                    boxes = result_item.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confs = result_item.boxes.conf.cpu().numpy()
                    cls_ids = result_item.boxes.cls.cpu().numpy()
                    
                    # Apply confidence threshold
                    valid_indices = confs >= 0.25  # Lowered confidence threshold to capture more predictions
                    boxes = boxes[valid_indices]
                    cls_ids = cls_ids[valid_indices].astype(int)
                    confs = confs[valid_indices]
                    
                    # Get image dimensions for coordinate conversion
                    img_height, img_width = img.shape[0], img.shape[1]
                    
                    for i in range(len(boxes)):
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
            
            # Add to all detections and ground truths
            for pred_box in pred_boxes:
                class_name = pred_box['class_name']
                if class_name in all_detections:
                    # Format: [image_id, confidence, x_center, y_center, width, height]
                    all_detections[class_name].append({
                        'image_id': img_idx,
                        'confidence': pred_box['confidence'],
                        'bbox': pred_box['bbox'],
                        'is_matched': False  # Will be set to True when matched with GT
                    })
            
            for gt_box in gt_boxes:
                class_name = gt_box['class_name']
                if class_name in all_ground_truths:
                    # Format: [image_id, x_center, y_center, width, height, is_matched]
                    all_ground_truths[class_name].append({
                        'image_id': img_idx,
                        'bbox': gt_box['bbox'],
                        'is_matched': False  # Will be set to True when matched with detection
                    })
        
        # Calculate mAP at different IoU thresholds
        iou_thresholds = [0.5, 0.75, 0.9]
        per_iou_ap = {}
        
        for iou_thresh in iou_thresholds:
            ap, per_class_ap = self._calculate_ap_at_iou_threshold(
                all_detections, all_ground_truths, class_names, iou_thresh
            )
            per_iou_ap[str(iou_thresh)] = ap
        
        # Calculate overall AP
        overall_ap, per_class_ap = self._calculate_ap_at_iou_threshold(
            all_detections, all_ground_truths, class_names, 0.5
        )
        
        # Calculate Average Recall (AR)
        overall_ar, per_class_ar = self._calculate_ar_at_iou_threshold(
            all_detections, all_ground_truths, class_names, 0.5
        )
        
        # Calculate bounding box regression metrics
        mean_iou, per_class_mean_iou = self._calculate_mean_iou(
            all_detections, all_ground_truths, class_names
        )
        
        center_point_accuracy, per_class_center_point_accuracy = self._calculate_center_point_accuracy(
            all_detections, all_ground_truths, class_names
        )
        
        bbox_size_error, per_class_bbox_size_error = self._calculate_bbox_size_error(
            all_detections, all_ground_truths, class_names
        )
        
        # Calculate true positives, false positives, false negatives
        tp_per_class, fp_per_class, fn_per_class = self._calculate_tp_fp_fn(
            all_detections, all_ground_truths, class_names
        )
        
        # Calculate overall metrics
        total_tp = sum(tp_per_class.values())
        total_fp = sum(fp_per_class.values())
        total_fn = sum(fn_per_class.values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_fpr = total_fp / (total_fp + total_fn) if (total_fp + total_fn) > 0 else 0  # False positive rate
        
        # Update metrics dictionary
        self.metrics["object_detection_metrics"]["mAP"]["overall"] = np.mean(list(per_iou_ap.values()))
        self.metrics["object_detection_metrics"]["mAP"]["per_iou"] = per_iou_ap
        self.metrics["object_detection_metrics"]["mAP"]["per_class"] = per_class_ap
        
        self.metrics["object_detection_metrics"]["AP"]["overall"] = overall_ap
        self.metrics["object_detection_metrics"]["AP"]["per_class"] = per_class_ap
        
        self.metrics["object_detection_metrics"]["AR"]["overall"] = overall_ar
        self.metrics["object_detection_metrics"]["AR"]["per_class"] = per_class_ar
        
        self.metrics["object_detection_metrics"]["true_positives"]["overall"] = total_tp
        self.metrics["object_detection_metrics"]["true_positives"]["per_class"] = tp_per_class
        self.metrics["object_detection_metrics"]["false_positives"]["overall"] = total_fp
        self.metrics["object_detection_metrics"]["false_positives"]["per_class"] = fp_per_class
        self.metrics["object_detection_metrics"]["false_negatives"]["overall"] = total_fn
        self.metrics["object_detection_metrics"]["false_negatives"]["per_class"] = fn_per_class
        
        self.metrics["object_detection_metrics"]["mean_iou"]["overall"] = mean_iou
        self.metrics["object_detection_metrics"]["mean_iou"]["per_class"] = per_class_mean_iou
        
        self.metrics["object_detection_metrics"]["center_point_accuracy"]["overall"] = center_point_accuracy
        self.metrics["object_detection_metrics"]["center_point_accuracy"]["per_class"] = per_class_center_point_accuracy
        
        self.metrics["object_detection_metrics"]["bbox_size_error"]["overall"] = bbox_size_error
        self.metrics["object_detection_metrics"]["bbox_size_error"]["per_class"] = per_class_bbox_size_error
        
        self.metrics["object_detection_metrics"]["false_positive_rate"]["overall"] = overall_fpr
        self.metrics["object_detection_metrics"]["false_positive_rate"]["per_class"] = {
            class_name: fp_per_class[class_name] / (fp_per_class[class_name] + fn_per_class[class_name])
            if (fp_per_class[class_name] + fn_per_class[class_name]) > 0 else 0
            for class_name in class_names
        }
        
        print(f"DEBUG: Object detection metrics calculated:")
        print(f" mAP: {self.metrics['object_detection_metrics']['mAP']['overall']:.3f}")
        print(f" AP per class: {per_class_ap}")
        print(f"  AR: {overall_ar:.3f}")
        print(f" Mean IoU: {mean_iou:.3f}")
        print(f"  TP/FP/FN: {total_tp}/{total_fp}/{total_fn}")

    def _calculate_ap_at_iou_threshold(self, all_detections, all_ground_truths, class_names, iou_threshold):
        """
        Calculate Average Precision at a specific IoU threshold.
        
        Args:
            all_detections: Dictionary of detections per class
            all_ground_truths: Dictionary of ground truths per class
            class_names: List of class names
            iou_threshold: IoU threshold for matching
            
        Returns:
            tuple: (overall_ap, per_class_ap)
        """
        per_class_ap = {}
        
        for class_name in class_names:
            detections = all_detections.get(class_name, [])
            ground_truths = all_ground_truths.get(class_name, [])
            
            if not detections or not ground_truths:
                per_class_ap[class_name] = 0.0
                continue
            
            # Sort detections by confidence in descending order
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # Initialize matched flags for ground truths
            gt_matched = [False] * len(ground_truths)
            
            # Calculate precision and recall at each detection point
            tp_list = []
            fp_list = []
            
            for detection in detections:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth for this detection
                for gt_idx, gt in enumerate(ground_truths):
                    if gt_matched[gt_idx] or gt['image_id'] != detection['image_id']:
                        continue  # Skip already matched GT or different image
                    
                    iou = self._calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx != -1:
                    # True positive - matched with ground truth
                    if not gt_matched[best_gt_idx]:  # Only count if not already matched
                        tp_list.append(1)
                        fp_list.append(0)
                        gt_matched[best_gt_idx] = True
                    else:
                        # Already matched, so this is a false positive
                        tp_list.append(0)
                        fp_list.append(1)
                else:
                    # False positive - no matching ground truth
                    tp_list.append(0)
                    fp_list.append(1)
            
            # Calculate cumulative true positives and false positives
            cum_tp = np.cumsum(tp_list)
            cum_fp = np.cumsum(fp_list)
            
            # Calculate precision and recall
            total_detections = len(detections)
            total_gt_class = len(ground_truths)
            
            recalls = cum_tp / total_gt_class if total_gt_class > 0 else np.zeros(len(cum_tp))
            precisions = cum_tp / (cum_tp + cum_fp) if total_detections > 0 else np.zeros(len(cum_tp))
            
            # Calculate Average Precision using the 11-point interpolation method
            if len(recalls) > 0 and len(precisions) > 0:
                # Interpolate precision at 11 recall levels (0.0, 0.1, ..., 1.0)
                ap = 0.0
                for t in np.arange(0.0, 1.1, 0.1):
                    mask = recalls >= t
                    if np.any(mask):
                        p_at_t = np.max(precisions[mask])
                        ap += p_at_t
                ap /= 11.0  # Average over 11 recall levels
                per_class_ap[class_name] = ap
            else:
                per_class_ap[class_name] = 0.0
        
        # Calculate overall AP
        overall_ap = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
        
        return overall_ap, per_class_ap

    def _calculate_ar_at_iou_threshold(self, all_detections, all_ground_truths, class_names, iou_threshold):
        """
        Calculate Average Recall at a specific IoU threshold.
        
        Args:
            all_detections: Dictionary of detections per class
            all_ground_truths: Dictionary of ground truths per class
            class_names: List of class names
            iou_threshold: IoU threshold for matching
            
        Returns:
            tuple: (overall_ar, per_class_ar)
        """
        per_class_ar = {}
        
        for class_name in class_names:
            detections = all_detections.get(class_name, [])
            ground_truths = all_ground_truths.get(class_name, [])
            
            if not ground_truths:
                per_class_ar[class_name] = 0.0
                continue
            
            if not detections:
                per_class_ar[class_name] = 0.0
                continue
            
            # Sort detections by confidence in descending order
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # Initialize matched flags for ground truths
            gt_matched = [False] * len(ground_truths)
            
            tp = 0  # True positives
            
            for detection in detections:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth for this detection
                for gt_idx, gt in enumerate(ground_truths):
                    if gt_matched[gt_idx] or gt['image_id'] != detection['image_id']:
                        continue  # Skip already matched GT or different image
                    
                    iou = self._calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx != -1:
                    # True positive - matched with ground truth
                    gt_matched[best_gt_idx] = True
                    tp += 1
            
            # Calculate recall for this class
            total_gt_class = len(ground_truths)
            recall = tp / total_gt_class if total_gt_class > 0 else 0
            
            per_class_ar[class_name] = recall
       
        # Calculate overall AR
        overall_ar = np.mean(list(per_class_ar.values())) if per_class_ar else 0.0
       
        return overall_ar, per_class_ar

    def _calculate_mean_iou(self, all_detections, all_ground_truths, class_names):
        """
        Calculate mean IoU between matched detections and ground truths.
        
        Args:
            all_detections: Dictionary of detections per class
            all_ground_truths: Dictionary of ground truths per class
            class_names: List of class names
            
        Returns:
            tuple: (overall_mean_iou, per_class_mean_iou)
        """
        per_class_ious = {class_name: [] for class_name in class_names}
        
        for class_name in class_names:
            detections = all_detections.get(class_name, [])
            ground_truths = all_ground_truths.get(class_name, [])
            
            if not detections or not ground_truths:
                continue
            
            # Sort detections by confidence in descending order
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # Initialize matched flags for ground truths
            gt_matched = [False] * len(ground_truths)
            
            for detection in detections:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth for this detection
                for gt_idx, gt in enumerate(ground_truths):
                    if gt_matched[gt_idx] or gt['image_id'] != detection['image_id']:
                        continue  # Skip already matched GT or different image
                    
                    iou = self._calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx != -1 and best_iou > 0:
                    # Add IoU to the list for this class
                    per_class_ious[class_name].append(best_iou)
                    gt_matched[best_gt_idx] = True  # Mark as matched to avoid duplicate matching
        
        # Calculate mean IoU per class
        per_class_mean_iou = {}
        all_ious = []
        
        for class_name in class_names:
            if per_class_ious[class_name]:
                per_class_mean_iou[class_name] = np.mean(per_class_ious[class_name])
                all_ious.extend(per_class_ious[class_name])
            else:
                per_class_mean_iou[class_name] = 0.0
        
        # Calculate overall mean IoU
        overall_mean_iou = np.mean(all_ious) if all_ious else 0.0
        
        return overall_mean_iou, per_class_mean_iou

    def _calculate_center_point_accuracy(self, all_detections, all_ground_truths, class_names):
        """
        Calculate center point accuracy between matched detections and ground truths.
        
        Args:
            all_detections: Dictionary of detections per class
            all_ground_truths: Dictionary of ground truths per class
            class_names: List of class names
            
        Returns:
            tuple: (overall_center_point_accuracy, per_class_center_point_accuracy)
        """
        per_class_distances = {class_name: [] for class_name in class_names}
        
        for class_name in class_names:
            detections = all_detections.get(class_name, [])
            ground_truths = all_ground_truths.get(class_name, [])
            
            if not detections or not ground_truths:
                continue
            
            # Sort detections by confidence in descending order
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # Initialize matched flags for ground truths
            gt_matched = [False] * len(ground_truths)
            
            for detection in detections:
                best_iou = 0
                best_gt_idx = -1
                best_distance = float('inf')
                
                # Find best matching ground truth for this detection
                for gt_idx, gt in enumerate(ground_truths):
                    if gt_matched[gt_idx] or gt['image_id'] != detection['image_id']:
                        continue  # Skip already matched GT or different image
                    
                    iou = self._calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
                        # Calculate center point distance
                        det_center_x, det_center_y = detection['bbox'][0], detection['bbox'][1]
                        gt_center_x, gt_center_y = gt['bbox'][0], gt['bbox'][1]
                        distance = np.sqrt((det_center_x - gt_center_x)**2 + (det_center_y - gt_center_y)**2)
                        best_distance = distance
                
                if best_gt_idx != -1:
                    # Add distance to the list for this class (lower is better, so accuracy is 1 - distance)
                    per_class_distances[class_name].append(1 - min(best_distance, 1.0))  # Clamp to [0, 1]
                    gt_matched[best_gt_idx] = True  # Mark as matched to avoid duplicate matching
        
        # Calculate center point accuracy per class
        per_class_center_point_accuracy = {}
        all_accuracies = []
        
        for class_name in class_names:
            if per_class_distances[class_name]:
                per_class_center_point_accuracy[class_name] = np.mean(per_class_distances[class_name])
                all_accuracies.extend(per_class_distances[class_name])
            else:
                per_class_center_point_accuracy[class_name] = 0.0
        
        # Calculate overall center point accuracy
        overall_center_point_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        
        return overall_center_point_accuracy, per_class_center_point_accuracy

    def _calculate_bbox_size_error(self, all_detections, all_ground_truths, class_names):
        """
        Calculate bounding box size error between matched detections and ground truths.
        
        Args:
            all_detections: Dictionary of detections per class
            all_ground_truths: Dictionary of ground truths per class
            class_names: List of class names
            
        Returns:
            tuple: (overall_bbox_size_error, per_class_bbox_size_error)
        """
        per_class_errors = {class_name: [] for class_name in class_names}
        
        for class_name in class_names:
            detections = all_detections.get(class_name, [])
            ground_truths = all_ground_truths.get(class_name, [])
            
            if not detections or not ground_truths:
                continue
            
            # Sort detections by confidence in descending order
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # Initialize matched flags for ground truths
            gt_matched = [False] * len(ground_truths)
            
            for detection in detections:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth for this detection
                for gt_idx, gt in enumerate(ground_truths):
                    if gt_matched[gt_idx] or gt['image_id'] != detection['image_id']:
                        continue # Skip already matched GT or different image
                    
                    iou = self._calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx != -1:
                    # Calculate size error (difference in width and height)
                    det_width, det_height = detection['bbox'][2], detection['bbox'][3]
                    gt_width, gt_height = gt['bbox'][2], gt['bbox'][3]
                    
                    width_error = abs(det_width - gt_width)
                    height_error = abs(det_height - gt_height)
                    size_error = (width_error + height_error) / 2  # Average error
                    
                    per_class_errors[class_name].append(size_error)
                    gt_matched[best_gt_idx] = True # Mark as matched to avoid duplicate matching
        
        # Calculate bbox size error per class
        per_class_bbox_size_error = {}
        all_errors = []
        
        for class_name in class_names:
            if per_class_errors[class_name]:
                per_class_bbox_size_error[class_name] = np.mean(per_class_errors[class_name])
                all_errors.extend(per_class_errors[class_name])
            else:
                per_class_bbox_size_error[class_name] = 0.0
        
        # Calculate overall bbox size error
        overall_bbox_size_error = np.mean(all_errors) if all_errors else 0.0
        
        return overall_bbox_size_error, per_class_bbox_size_error

    def _calculate_tp_fp_fn(self, all_detections, all_ground_truths, class_names):
        """
        Calculate true positives, false positives, and false negatives per class.
        
        Args:
            all_detections: Dictionary of detections per class
            all_ground_truths: Dictionary of ground truths per class
            class_names: List of class names
            
        Returns:
            tuple: (tp_per_class, fp_per_class, fn_per_class)
        """
        tp_per_class = {}
        fp_per_class = {}
        fn_per_class = {}
        
        for class_name in class_names:
            detections = all_detections.get(class_name, [])
            ground_truths = all_ground_truths.get(class_name, [])
            
            if not detections and not ground_truths:
                tp_per_class[class_name] = 0
                fp_per_class[class_name] = 0
                fn_per_class[class_name] = 0
                continue
            
            if not detections:
                tp_per_class[class_name] = 0
                fp_per_class[class_name] = 0
                fn_per_class[class_name] = len(ground_truths)
                continue
            
            if not ground_truths:
                tp_per_class[class_name] = 0
                fp_per_class[class_name] = len(detections)
                fn_per_class[class_name] = 0
                continue
            
            # Sort detections by confidence in descending order
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # Initialize matched flags for ground truths
            gt_matched = [False] * len(ground_truths)
            
            tp = 0  # True positives
            fp = 0 # False positives
            
            for detection in detections:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth for this detection
                for gt_idx, gt in enumerate(ground_truths):
                    if gt_matched[gt_idx] or gt['image_id'] != detection['image_id']:
                        continue  # Skip already matched GT or different image
                    
                    iou = self._calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou and iou >= 0.5:  # Standard IoU threshold
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx != -1:
                    # True positive - matched with ground truth
                    gt_matched[best_gt_idx] = True
                    tp += 1
                else:
                    # False positive - no matching ground truth
                    fp += 1
            
            # False negatives are ground truths that were not matched
            fn = sum(1 for matched in gt_matched if not matched)
            
            tp_per_class[class_name] = tp
            fp_per_class[class_name] = fp
            fn_per_class[class_name] = fn
        
        return tp_per_class, fp_per_class, fn_per_class

    def _compute_classification_metrics(self, images_path, labels_path, class_names):
        """
        Compute classification metrics by comparing model predictions to ground truth labels.
        This method handles image classification tasks where each image has a single class.
        
        Args:
            images_path (str): Path to test images
            labels_path (str): Path to ground truth labels
            class_names (list): List of class names
            
        Returns:
            tuple: (precision, recall, f1_score, accuracy, per_class_metrics)
        """
        # For classification, we need to match each image to its ground truth class
        # This is different from detection where we have bounding boxes
        
        # Initialize counters
        y_true = []
        y_pred = []
        
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            
            # Run inference
            results = self.model(img)
            
            # Get predicted class
            pred_class_id = None
            pred_confidence = 0
            
            if results and len(results) > 0:
                result_item = results[0] if isinstance(results, list) else results
                
                if hasattr(result_item, 'probs') and result_item.probs is not None:
                    # For classification models
                    pred_class_id = int(result_item.probs.top1)
                    pred_confidence = float(result_item.probs.top1conf)
                elif hasattr(result_item, 'boxes') and result_item.boxes is not None and len(result_item.boxes) > 0:
                    # For detection models, get the highest confidence detection
                    confs = result_item.boxes.conf.cpu().numpy()
                    class_ids = result_item.boxes.cls.cpu().numpy()
                    max_idx = np.argmax(confs)
                    pred_class_id = int(class_ids[max_idx])
                    pred_confidence = float(confs[max_idx])
            
            # Get ground truth class from label file
            # For classification, labels might be in a different format
            gt_class_id = self._get_ground_truth_class_id(img_file, labels_path, class_names)
            
            if pred_class_id is not None and gt_class_id is not None:
                y_true.append(gt_class_id)
                y_pred.append(pred_class_id)
        
        if len(y_true) == 0:
            print("DEBUG: No valid samples found for classification metrics calculation")
            # Return default values
            per_class_metrics = {}
            for class_name in class_names:
                per_class_metrics[class_name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "accuracy": 0.0
                }
            return 0.0, 0.0, 0.0, 0.0, per_class_metrics
        
        # Calculate overall metrics using sklearn
        precision, recall, f1_score, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                per_class_metrics[class_name] = {
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1_score": float(f1_per_class[i]),
                    "accuracy": 0.0  # Accuracy per class is not standard, so set to 0
                }
            else:
                per_class_metrics[class_name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "accuracy": 0.0
                }
        
        return float(precision), float(recall), float(f1_score), float(accuracy), per_class_metrics

    def _get_ground_truth_class_id(self, img_file, labels_path, class_names):
        """
        Get ground truth class ID for an image file.
        This is a helper method to determine the ground truth for classification tasks.
        """
        # For classification, ground truth might be encoded in the filename or in separate label files
        img_name = os.path.splitext(img_file)[0]
        
        # Try to determine class from directory structure (e.g., images/class_name/img.jpg)
        parent_dir = os.path.basename(os.path.dirname(os.path.join(labels_path, img_file)))
        for i, class_name in enumerate(class_names):
            if parent_dir.lower() == class_name.lower():
                return i
        
        # If not found in directory, try to match based on filename pattern
        for i, class_name in enumerate(class_names):
            if class_name.lower() in img_name.lower():
                return i
        
        # If no match found, return None
        return None
        
    def _load_segmentation_ground_truth(self, image_path, labels_path, img_size=(640, 640)):
        """
        Load ground truth segmentation masks from YOLO format.
        In YOLO segmentation format, the labels contain normalized polygon coordinates.
        
        Args:
            image_path (str): Path to the image file
            labels_path (str): Path to the labels directory
            img_size (tuple): Size of the image (width, height) for denormalization
            
        Returns:
            numpy array: Ground truth segmentation mask
        """
        import os
        import sys
        from pathlib import Path
        
        # Add the src directory to the path to ensure proper imports
        src_dir = Path(__file__).parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            
        from image_utils import yolo_to_mask_format
        
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

    def _compute_segmentation_metrics(self, images_path, labels_path, class_names):
         """
         Compute segmentation metrics by comparing predicted masks to ground truth masks.
         Calculates IoU, Dice coefficient, pixel accuracy, and mean absolute error.
         Uses YOLO format for ground truth masks.
         
         Args:
             images_path (str): Path to test images
             labels_path (str): Path to ground truth labels in YOLO format
             class_names (list): List of class names
             
         Returns:
             tuple: (iou_score, dice_score, pixel_accuracy, mae, per_class_metrics)
         """
         # Initialize metrics accumulators
         total_iou = 0
         total_dice = 0
         total_pixel_accuracy = 0
         total_mae = 0
         num_samples = 0
         
         # Per-class metrics
         per_class_metrics = {}
         for class_name in class_names:
             per_class_metrics[class_name] = {
                 "iou": [],
                 "dice": [],
                 "pixel_accuracy": [],
                 "mae": []
             }
         
         # Get all image files
         image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
         
         print(f"DEBUG: Processing {len(image_files)} images for segmentation metrics")
         print(f"DEBUG: Class names: {class_names}")
         
         for img_idx, img_file in enumerate(image_files):
             # Load image and corresponding ground truth mask in YOLO format
             img_path = os.path.join(images_path, img_file)
             
             # Load ground truth mask from YOLO format labels
             gt_mask = self._load_segmentation_ground_truth(img_path, labels_path)
             
             if gt_mask is None or gt_mask.size == 0:
                 print(f"DEBUG: Could not load ground truth mask for {img_file}, skipping")
                 continue
             
             # Run inference to get predicted masks
             results = self.model(img_path)
             
             # Extract predicted masks
             pred_masks = []  # Store all predicted masks for multi-object segmentation
             if results and len(results) > 0:
                 result_item = results[0] if isinstance(results, list) else results
                 
                 if hasattr(result_item, 'masks') and result_item.masks is not None:
                     if len(result_item.masks) > 0:
                         # Get all masks and convert to numpy arrays
                         for mask_tensor in result_item.masks.data:
                             pred_mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                             # Resize prediction mask to match ground truth dimensions if needed
                             if pred_mask.shape != gt_mask.shape:
                                 pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                             pred_masks.append(pred_mask)
                     else:
                         # No masks predicted, create empty mask
                         pred_masks = [np.zeros_like(gt_mask)]
                 else:
                     # No masks attribute, create empty mask
                     pred_masks = [np.zeros_like(gt_mask)]
             else:
                 # No results from model, create empty mask
                 pred_masks = [np.zeros_like(gt_mask)]
             
             # Create a combined prediction mask from all individual masks
             combined_pred_mask = np.zeros_like(gt_mask)
             for pred_mask in pred_masks:
                 combined_pred_mask = np.logical_or(combined_pred_mask, pred_mask).astype(np.uint8)
             
             # Calculate metrics for this sample
             # Convert to binary masks (0 or 1 for foreground/background)
             gt_binary = (gt_mask > 0).astype(np.uint8)
             pred_binary = combined_pred_mask
             
             # Calculate Intersection over Union (IoU)
             intersection = np.logical_and(gt_binary, pred_binary).sum()
             union = np.logical_or(gt_binary, pred_binary).sum()
             iou = intersection / union if union > 0 else 0
             
             # Calculate Dice coefficient
             dice = (2 * intersection) / (gt_binary.sum() + pred_binary.sum()) if (gt_binary.sum() + pred_binary.sum()) > 0 else 0
             
             # Calculate pixel accuracy
             pixel_accuracy = np.sum(gt_binary == pred_binary) / gt_binary.size
             
             # Calculate mean absolute error (MAE) between masks
             mae = np.mean(np.abs(gt_binary.astype(float) - pred_binary.astype(float)))
             
             # Accumulate metrics
             total_iou += iou
             total_dice += dice
             total_pixel_accuracy += pixel_accuracy
             total_mae += mae
             num_samples += 1
             
             # Add to per-class metrics (for segmentation, we often just track overall metrics)
             # In case of multi-class segmentation, we would need to handle each class separately
             for class_name in class_names:
                 per_class_metrics[class_name]["iou"].append(iou)
                 per_class_metrics[class_name]["dice"].append(dice)
                 per_class_metrics[class_name]["pixel_accuracy"].append(pixel_accuracy)
                 per_class_metrics[class_name]["mae"].append(mae)
             
             print(f"DEBUG: Image {img_idx}: IoU={iou:.3f}, Dice={dice:.3f}, Pixel Acc={pixel_accuracy:.3f}, MAE={mae:.3f}")
         
         if num_samples == 0:
             print("DEBUG: No valid samples found for segmentation metrics calculation")
             return 0.0, 0.0, 0.0, 0.0, per_class_metrics
         
         # Calculate average metrics across all samples
         avg_iou = total_iou / num_samples
         avg_dice = total_dice / num_samples
         avg_pixel_accuracy = total_pixel_accuracy / num_samples
         avg_mae = total_mae / num_samples
         
         # Calculate per-class averages
         for class_name in class_names:
             class_metrics = per_class_metrics[class_name]
             per_class_metrics[class_name] = {
                 "iou": float(np.mean(class_metrics["iou"])) if class_metrics["iou"] else 0.0,
                 "dice": float(np.mean(class_metrics["dice"])) if class_metrics["dice"] else 0.0,
                 "pixel_accuracy": float(np.mean(class_metrics["pixel_accuracy"])) if class_metrics["pixel_accuracy"] else 0.0,
                 "mae": float(np.mean(class_metrics["mae"])) if class_metrics["mae"] else 0.0
             }
         
         print(f"DEBUG: Average segmentation metrics - IoU: {avg_iou:.3f}, Dice: {avg_dice:.3f}, Pixel Acc: {avg_pixel_accuracy:.3f}, MAE: {avg_mae:.3f}")
         
         return float(avg_iou), float(avg_dice), float(avg_pixel_accuracy), float(avg_mae), per_class_metrics

     
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
        # Check KPI targets for classification/detection metrics
        kpi_results = {
            "recall_for_problematic_cases_met": self.metrics["classification_metrics"]["overall"]["recall"] >= 0.80,
            "precision_for_marked_items_met": self.metrics["classification_metrics"]["overall"]["precision"] >= 0.85,
            "unnecessary_suggestions_met": (1 - self.metrics["classification_metrics"]["overall"]["precision"]) <= 0.10
        }
        
        # Add KPI targets for segmentation metrics if they exist
        if "segmentation_metrics" in self.metrics:
            seg_kpi_results = {
                "segmentation_iou_met": self.metrics["segmentation_metrics"]["overall"]["iou"] >= 0.70,
                "segmentation_dice_met": self.metrics["segmentation_metrics"]["overall"]["dice_coefficient"] >= 0.75,
                "segmentation_pixel_accuracy_met": self.metrics["segmentation_metrics"]["overall"]["pixel_accuracy"] >= 0.90
            }
            kpi_results.update(seg_kpi_results)
        
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
        
        # Determine task type based on model type and calculate appropriate metrics
        print("Determining task type and calculating appropriate metrics...")
        
        # Load model configuration to determine the task type
        model_classes = self._get_classes_from_model_config(model_config_path, "detection")  # Default to detection
        
        # Check if the model is a segmentation model by checking its capabilities
        task_type = "detection"  # Default
        
        # If the model has segmentation capabilities, use segmentation task type
        if hasattr(self.model, 'task') and self.model.task == 'segment':
            task_type = "segmentation"
        elif hasattr(self.model, 'names') and any('defect' in name.lower() or 'mask' in name.lower() for name in self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names:
            task_type = "segmentation"
        else:
            # Try to infer from model path or name
            model_path_lower = self.model_path.lower()
            if 'seg' in model_path_lower or 'segment' in model_path_lower:
                task_type = "segmentation"
        
        print(f"DEBUG: Determined task type: {task_type}")
        
        self.calculate_classification_metrics(test_images_path, task_type=task_type, model_config_path=model_config_path)
        
        # For segmentation models, run additional segmentation-specific validation
        if task_type == "segmentation":
            print("Running segmentation-specific validation...")
            # Load the dataset config to get the labels path
            with open(self.data_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            # Get the test labels path from the YAML config
            base_path = os.path.dirname(self.data_path)
            test_labels_rel_path = dataset_config.get('test', 'test/images').replace('images', 'labels')  # Convert images path to labels path
            test_labels_path = os.path.join(base_path, test_labels_rel_path)
            
            # If the path starts with '../' or './', resolve it properly
            test_labels_path = os.path.abspath(test_labels_path)
            
            # If the resolved path doesn't exist, try alternative paths
            if not os.path.exists(test_labels_path):
                # Try alternative paths for labels
                alt_paths = [
                    os.path.join(base_path, 'test', 'labels'),
                    os.path.join(base_path, 'labels', 'test'),
                    os.path.join(base_path, test_labels_rel_path),  # Try with original relative path
                    test_images_path.replace('images', 'labels') if 'images' in test_images_path else test_images_path
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        test_labels_path = alt_path
                        break
            
            if os.path.exists(test_labels_path):
                print(f"Running segmentation validation with labels from: {test_labels_path}")
                self.run_segmentation_validation(test_images_path, test_labels_path)
            else:
                print(f"Warning: Labels path does not exist: {test_labels_path}. Skipping segmentation-specific validation.")
        
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
    
    def run_segmentation_validation(self, images_path, labels_path, output_dir=None, num_samples_for_viz=5):
        """
        Run comprehensive segmentation validation with visualization.
        
        Args:
            images_path (str): Path to test images
            labels_path (str): Path to ground truth labels in YOLO format
            output_dir (str): Directory to save visualizations (defaults to validation_visualizations subdirectory)
            num_samples_for_viz (int): Number of samples to visualize (default: 5)
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "validation_visualizations")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get class names for the model
        model_classes = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
        
        print(f"Running segmentation validation on {images_path} with labels from {labels_path}")
        print(f"Model classes: {model_classes}")
        
        # Calculate segmentation metrics
        iou_score, dice_score, pixel_accuracy, mae, per_class_metrics = self._compute_segmentation_metrics(
            images_path, labels_path, model_classes
        )
        
        # Update metrics
        self.metrics["segmentation_metrics"]["overall"]["iou"] = iou_score
        self.metrics["segmentation_metrics"]["overall"]["dice_coefficient"] = dice_score
        self.metrics["segmentation_metrics"]["overall"]["pixel_accuracy"] = pixel_accuracy
        self.metrics["segmentation_metrics"]["overall"]["mean_absolute_error"] = mae
        self.metrics["segmentation_metrics"]["per_class"] = per_class_metrics
        
        print(f"Segmentation validation results:")
        print(f"  IoU: {iou_score:.3f}")
        print(f"  Dice: {dice_score:.3f}")
        print(f"  Pixel Accuracy: {pixel_accuracy:.3f}")
        print(f"  MAE: {mae:.3f}")
        
        # Generate visualizations for a subset of samples
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        viz_count = 0
        
        for img_file in image_files:
            if viz_count >= num_samples_for_viz:
                break
                
            img_path = os.path.join(images_path, img_file)
            
            # Load ground truth mask
            gt_mask = self._load_segmentation_ground_truth(img_path, labels_path)
            
            # Run inference to get predicted mask
            results = self.model(img_path)
            pred_masks = []
            
            if results and len(results) > 0:
                result_item = results[0] if isinstance(results, list) else results
                
                if hasattr(result_item, 'masks') and result_item.masks is not None:
                    if len(result_item.masks) > 0:
                        for mask_tensor in result_item.masks.data:
                            pred_mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                            if pred_mask.shape != gt_mask.shape:
                                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                            pred_masks.append(pred_mask)
                    else:
                        pred_masks = [np.zeros_like(gt_mask)]
                else:
                    pred_masks = [np.zeros_like(gt_mask)]
            else:
                pred_masks = [np.zeros_like(gt_mask)]
            
            # Create combined prediction mask
            combined_pred_mask = np.zeros_like(gt_mask)
            for pred_mask in pred_masks:
                combined_pred_mask = np.logical_or(combined_pred_mask, pred_mask).astype(np.uint8)
            
            # Create visualization
            viz_filename = f"seg_viz_{os.path.splitext(img_file)[0]}.png"
            viz_path = os.path.join(output_dir, viz_filename)
            self.visualize_segmentation_results(img_path, gt_mask, combined_pred_mask, viz_path)
            
            viz_count += 1
        
        print(f"Visualizations saved to {output_dir}")
        
        return {
            "iou": iou_score,
            "dice": dice_score,
            "pixel_accuracy": pixel_accuracy,
            "mae": mae,
            "per_class_metrics": per_class_metrics
        }
    
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
    
    def visualize_segmentation_results(self, image_path, gt_mask, pred_mask, output_path, alpha=0.5):
        """
        Create a visualization showing ground truth and predicted segmentation masks overlaid on the image.
        
        Args:
            image_path (str): Path to the original image
            gt_mask (numpy array): Ground truth segmentation mask
            pred_mask (numpy array): Predicted segmentation mask
            output_path (str): Path to save the visualization
            alpha (float): Transparency for mask overlays (0.0-1.0)
        """
        import matplotlib.pyplot as plt
        
        # Load the original image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create the visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask overlay
        gt_overlay = img_rgb.copy()
        gt_overlay[gt_mask > 0] = [255, 0, 0]  # Red for ground truth
        axes[1].imshow(cv2.addWeighted(img_rgb, 1 - alpha, gt_overlay, alpha, 0))
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Predicted mask overlay
        pred_overlay = img_rgb.copy()
        pred_overlay[pred_mask > 0] = [0, 255, 0]  # Green for prediction
        axes[2].imshow(cv2.addWeighted(img_rgb, 1 - alpha, pred_overlay, alpha, 0))
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Segmentation visualization saved to {output_path}")

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