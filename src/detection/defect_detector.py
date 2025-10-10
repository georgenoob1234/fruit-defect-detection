"""
Defect Segmentation module for the Fruit Defect Detection System.

This module handles defect segmentation using a YOLO segmentation model.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class DefectSegmenter:
    """
    A class to handle defect segmentation using a YOLO segmentation model.
    """
    
    def __init__(self, model_path, confidence_threshold=0.5, target_classes=None):
        """
        Initialize the defect segmenter with the provided model.
        
        Args:
            model_path (str): Path to the YOLO segmentation model file
            confidence_threshold (float): Minimum confidence for segmentation (0.0-1.0)
            target_classes (list): List of defect classes the model should detect
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ['defect', 'defective']
        
        # Load the YOLO segmentation model
        self.model = YOLO(model_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Defect segmenter initialized with model: {model_path}")
        self.logger.info(f"Confidence threshold: {confidence_threshold}")
        self.logger.info(f"Target classes: {self.target_classes}")
    
    def segment_defects(self, fruit_region):
        """
        Segment defects in a fruit region.
        
        Args:
            fruit_region: Image region containing a single fruit (numpy array)
            
        Returns:
            tuple: (has_defects: bool, confidence: float, masks: list)
        """
        try:
            self.logger.info(f"Running defect segmentation on region with shape: {fruit_region.shape}")
            
            # Run inference on the fruit region
            results = self.model(fruit_region, conf=self.confidence_threshold)
            
            self.logger.info(f"Model inference completed, number of results: {len(results)}")
            
            # Process the results
            masks = []
            max_confidence = 0.0
            has_defects = False
            
            for result in results:
                self.logger.info(f"Processing result - has masks: {result.masks is not None}, has boxes: {result.boxes is not None}")
                
                if result.masks is not None:
                    # Process segmentation masks
                    for i, mask in enumerate(result.masks.data):
                        # Convert mask to numpy array
                        mask_np = mask.cpu().numpy()
                        
                        # Get the corresponding class and confidence
                        if result.boxes is not None and i < len(result.boxes):
                            box = result.boxes[i]
                            class_idx = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = result.names[class_idx]
                            
                            self.logger.info(f"Detection {i}: class='{class_name}', confidence={confidence:.2f}")
                            
                            # Check if this is a defect class
                            if class_name.lower() in self.target_classes:
                                has_defects = True
                                if confidence > max_confidence:
                                    max_confidence = confidence
                                
                                masks.append(mask_np)
                                self.logger.info(f"Found defect: class='{class_name}', confidence={confidence:.2f}")
                        else:
                            # If no boxes are available, assume all masks are defects
                            has_defects = True
                            masks.append(mask_np)
                            self.logger.info(f"Found mask without box, treating as defect")
            
            self.logger.info(f"Defect segmentation complete: {len(masks)} masks found, max confidence: {max_confidence:.2f}, has defects: {has_defects}")
            
            return has_defects, max_confidence, masks
            
        except Exception as e:
            self.logger.error(f"Error during defect segmentation: {e}")
            return False, 0.0, []
    
    def segment_fruit_defects(self, frame, bbox):
        """
        Extract fruit region from frame and segment defects.
        
        Args:
            frame: Input frame (numpy array)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            tuple: (has_defects: bool, confidence: float, masks: list, segmented_frame: numpy array)
        """
        try:
            # Extract the fruit region from the frame using the bounding box
            x1, y1, x2, y2 = map(int, bbox)  # Ensure bbox coordinates are integers
            self.logger.info(f"Extracting fruit region with bbox: [{x1}, {y1}, {x2}, {y2}]")
            
            fruit_region = frame[y1:y2, x1:x2]
            
            # Make sure the region is not empty
            if fruit_region.size == 0:
                self.logger.warning("Empty fruit region extracted, returning default segmentation")
                return False, 0.0, [], frame
            
            self.logger.info(f"Fruit region extracted with shape: {fruit_region.shape}")
            
            # Segment defects in the extracted region
            has_defects, confidence, masks = self.segment_defects(fruit_region)
            
            # Create a copy of the frame to draw segmentation on
            result_frame = frame.copy()
            
            # Draw segmentation masks on the frame if defects are found
            if has_defects and masks:
                for mask in masks:
                    # Resize mask to match the original fruit region size
                    resized_mask = cv2.resize(mask.astype(np.uint8),
                                            (x2 - x1, y2 - y1))
                    
                    # Create a colored overlay for the mask
                    colored_mask = np.zeros_like(result_frame[y1:y2, x1:x2])
                    colored_mask[:, :, 0] = resized_mask * 25  # Red channel for defects
                    
                    # Apply the mask to the result frame
                    result_frame[y1:y2, x1:x2] = np.where(
                        np.stack([resized_mask] * 3, axis=-1) > 0,
                        cv2.addWeighted(result_frame[y1:y2, x1:x2], 0.7, colored_mask, 0.3, 0),
                        result_frame[y1:y2, x1:x2]
                    )
            
            return has_defects, confidence, masks, result_frame
            
        except Exception as e:
            self.logger.error(f"Error extracting and segmenting fruit defects: {e}")
            return False, 0.0, [], frame