"""
Fruit Detector module for the Fruit Defect Detection System.

This module handles fruit detection using a YOLO model.
"""
import logging
import cv2
from pathlib import Path
from ultralytics import YOLO


class FruitDetector:
    """
    A class to handle fruit detection using a YOLO model.
    """
    
    def __init__(self, model_path, confidence_threshold=0.5, target_classes=None):
        """
        Initialize the fruit detector with the provided model.
        
        Args:
            model_path (str): Path to the YOLO model file
            confidence_threshold (float): Minimum confidence for detection (0.0-1.0)
            target_classes (list): List of fruit classes the model should detect
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ['apple', 'banana', 'tomato']
        
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Fruit detector initialized with model: {model_path}")
        self.logger.info(f"Confidence threshold: {confidence_threshold}")
        self.logger.info(f"Target classes: {self.target_classes}")
    
    def _is_valid_fruit_detection(self, bbox, frame_shape, class_name):
        """
        Validate if a detection is likely to be an actual fruit based on geometric properties.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            frame_shape: Shape of the input frame (height, width, channels)
            class_name: Name of the detected class
            
        Returns:
            bool: True if the detection is likely valid, False otherwise
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        frame_height, frame_width = frame_shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        
        # Calculate relative size (percentage of frame)
        bbox_area = width * height
        frame_area = frame_width * frame_height
        relative_size = bbox_area / frame_area
        
        # Validation criteria:
        # 1. Aspect ratio should be reasonable for fruits (not extremely wide or tall)
        # 2. Size should be within reasonable range for fruits
        # 3. Avoid detections that are too large (like full face)
        
        # For apples and tomatoes (round fruits), aspect ratio should be roughly 1:1
        if class_name.lower() in ['apple', 'tomato']:
            # Acceptable aspect ratio range for round fruits (0.5 to 2.0)
            # Made more permissive to catch apples that might appear oval-shaped in images
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                return False
        
        # For bananas (elongated fruit), aspect ratio should be different
        elif class_name.lower() == 'banana':
            # Bananas should have more elongated aspect ratio (typically 2:1 to 5:1)
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                return False
        
        # Size validation: fruit should be neither too small nor too large
        # Minimum size: at least 10x10 pixels (made more permissive)
        if width < 10 or height < 10:
            return False
        
        # Maximum size: should not be more than 50% of the frame area (made more permissive)
        if relative_size > 0.5:
            return False
            
        # Maximum dimension: should not exceed 60% of the frame in any direction (made more permissive)
        if width > frame_width * 0.6 or height > frame_height * 0.6:
            return False
        
        return True
    
    def detect_fruits(self, frame):
        """
        Detect fruits in the given frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            list: List of detection results, each containing:
                - fruit_class (str): The type of fruit detected
                - confidence (float): Confidence score (0.0-1.0)
                - bbox (list): Bounding box coordinates [x1, y1, x2, y2]
        """
        try:
            # Run inference on the frame
            results = self.model(frame, conf=self.confidence_threshold)
            
            detections = []
            
            # Process the results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class index and confidence
                        class_idx = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Get class name from model names
                        class_name = result.names[class_idx]
                        
                        # Check if the detected class is in our target classes
                        if class_name.lower() in [c.lower() for c in self.target_classes]:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [int(x1), int(y1), int(x2), int(y2)]
                            
                            # Validate the detection geometrically to avoid false positives
                            if self._is_valid_fruit_detection(bbox, frame.shape, class_name):
                                detection = {
                                    'fruit_class': class_name.lower(),
                                    'confidence': confidence,
                                    'bbox': bbox
                                }
                                
                                detections.append(detection)
                                
                                self.logger.debug(f"Fruit detected: {class_name} (confidence: {confidence:.2f}) at {bbox}")
                            else:
                                self.logger.debug(f"Discarded potential false positive: {class_name} (confidence: {confidence:.2f}) at {bbox}")
            
            self.logger.info(f"Detected {len(detections)} fruits in frame")
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during fruit detection: {e}")
            return []