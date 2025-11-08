"""
Detection Pipeline module for the Fruit Defect Detection System.

This module orchestrates the detection process, coordinating between
fruit detection and defect segmentation models.
"""
import logging
import cv2
import time
from datetime import datetime
from pathlib import Path
from src.detection.fruit_detector import FruitDetector
from src.detection.defect_detector import DefectSegmenter
from src.utils.ood_detector import OODDetector, apply_ood_filtering


class DetectionPipeline:
    """
    A class to coordinate the detection process between fruit detection 
    and defect segmentation models.
    """
    
    def __init__(self, fruit_detector, defect_segmenter, api_handler=None, ood_detector=None):
        """
        Initialize the detection pipeline with required components.
        
        Args:
            fruit_detector: Instance of FruitDetector
            defect_segmenter: Instance of DefectSegmenter (can be None if defect detection is disabled)
            api_handler: Optional APIHandler instance for sending data
            ood_detector: Optional OODDetector instance for out-of-distribution detection
        """
        self.fruit_detector = fruit_detector
        self.defect_segmenter = defect_segmenter
        self.api_handler = api_handler
        self.ood_detector = ood_detector
        
        # Check if defect detection is enabled based on whether defect_segmenter is provided
        self.defect_detection_enabled = defect_segmenter is not None
        self.ood_detection_enabled = ood_detector is not None and ood_detector.enable_ood_detection
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Detection pipeline initialized")
        self.logger.info(f"OOD detection enabled: {self.ood_detection_enabled}")
    
    def process_frame(self, frame):
        """
        Process a single frame to detect fruits and segment defects.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            list: List of detection results, each containing:
                - fruit_class (str): The type of fruit detected
                - is_defective (bool): Whether the fruit is defective
                - confidence (float): Overall confidence score (0.0-1.0)
                - bbox (list): Bounding box coordinates [x1, y1, x2, y2]
                - masks (list): List of segmentation masks for defects (if any)
        """
        try:
            # Step 1: Detect fruits in the frame
            fruit_detections = self.fruit_detector.detect_fruits(frame)
            
            self.logger.info(f"Found {len(fruit_detections)} fruit detections in frame")
            
            # Step 1.5: Apply OOD detection if enabled
            if self.ood_detection_enabled:
                self.logger.info("Applying OOD detection filtering...")
                filtered_detections = apply_ood_filtering(fruit_detections, self.ood_detector, frame)
                self.logger.info(f"After OOD filtering: {len(filtered_detections)} detections remain")
                
                # If frame is OOD, return empty results
                if len(filtered_detections) == 0 and len(fruit_detections) > 0:
                    self.logger.info("Frame detected as OOD, returning no detections")
                    return [], frame
            else:
                filtered_detections = fruit_detections
            
            results = []
            
            # Initialize segmented_frame to the original frame
            # This ensures the variable is always defined, even if no fruits are detected
            segmented_frame = frame
            
            # Step 2: For each detected fruit, segment defects (if defect detection is enabled)
            for i, detection in enumerate(filtered_detections):
                fruit_class = detection['fruit_class']
                fruit_confidence = detection['confidence']
                bbox = detection['bbox']
                
                self.logger.info(f"Fruit detection {i+1}: {fruit_class}, confidence: {fruit_confidence:.2f}, bbox: {bbox}")
                
                if self.defect_detection_enabled:
                    # Extract the fruit region and segment defects
                    try:
                        has_defects, defect_confidence, masks, segmented_frame = self.defect_segmenter.segment_fruit_defects(frame, bbox)
                        is_defective = has_defects
                        self.logger.info(f"Defect detection result: has_defects={has_defects}, defect_confidence={defect_confidence:.2f}, masks_count={len(masks)}")
                    except Exception as e:
                        self.logger.error(f"Error in defect segmentation: {e}")
                        is_defective = False
                        defect_confidence = 0.0
                        masks = []
                        # Continue with original frame if segmentation fails
                        segmented_frame = frame
                else:
                    # If defect detection is disabled, default to non-defective
                    is_defective = False
                    defect_confidence = 1.0  # Default confidence when no defect detection is performed
                    masks = []
                    self.logger.info("Defect detection is disabled, defaulting to non-defective")
                
                # Calculate overall confidence as a combination of fruit and defect confidence
                # Use the fruit detection confidence as the primary confidence
                overall_confidence = fruit_confidence
                
                result = {
                    'fruit_class': fruit_class,
                    'is_defective': is_defective,
                    'confidence': overall_confidence,
                    'bbox': bbox,
                    'masks': masks
                }
                
                results.append(result)
                
                self.logger.info(f"Processed detection: {fruit_class}, defective: {is_defective}, confidence: {overall_confidence:.2f}")
            
            self.logger.info(f"Frame processing complete: {len(results)} fruits processed")
            return results, segmented_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return [], frame
    
    def process_detection(self, detection_result, frame):
        """
        Process a single detection result, including photo capture, API notification, and Telegram notification.
        
        Args:
            detection_result (dict): Detection result with fruit_class, is_defective, confidence, bbox
            frame: The camera frame for photo capture if needed
        """
        fruit_class = detection_result['fruit_class']
        is_defective = detection_result['is_defective']
        confidence = detection_result['confidence']
        bbox = detection_result['bbox']
        
        # Prepare detection data
        detection_data = {
            'fruit_class': fruit_class,
            'is_defective': is_defective,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'bbox': bbox
        }
        
        self.logger.info(f"Processed detection: {fruit_class}, defective: {is_defective}, confidence: {confidence:.2f}")
        
        return detection_data