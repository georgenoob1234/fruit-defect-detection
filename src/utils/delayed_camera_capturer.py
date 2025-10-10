"""
Delayed Camera Capturer module for the Fruit Defect Detection System.

This module implements a camera capturer that only takes pictures after 
a consistent detection has been maintained for a specified duration.
"""
import time
import threading
from datetime import datetime
from pathlib import Path
import cv2
import logging


class DelayedCameraCapturer:
    """
    A class to handle delayed photo capture based on consistent detection.
    Only captures photos after a detection has been consistently present for a specified duration.
    """
    
    def __init__(self, delay_duration=1.0, image_save_path="captured_images/", image_format="jpg"):
        """
        Initialize the delayed camera capturer.
        
        Args:
            delay_duration (float): Duration in seconds that detection must be maintained before capturing
            image_save_path (str): Path to save captured images
            image_format (str): Image format for saved images
        """
        self.delay_duration = delay_duration
        self.image_save_path = image_save_path
        self.image_format = image_format
        
        # Create save directory if it doesn't exist
        Path(self.image_save_path).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Tracking variables
        self.current_detection_signature = None
        self.detection_start_time = None
        self.capture_timer = None
        self.pending_capture = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        self.logger.info(f"Delayed camera capturer initialized with delay: {delay_duration}s")
    
    def _get_detection_signature(self, detection_results):
        """
        Create a signature for the current detection results to identify if they are consistent.
        
        Args:
            detection_results (list): List of detection results
            
        Returns:
            str: A signature representing the current detection state
        """
        if not detection_results:
            return "no_detections"
        
        # Create a signature based on fruit classes, positions, and defect status
        signatures = []
        for result in detection_results:
            fruit_class = result['fruit_class']
            is_defective = result['is_defective']
            bbox = result['bbox']
            # Round bbox coordinates to reduce sensitivity to small movements
            rounded_bbox = tuple(round(coord) for coord in bbox)
            signature = f"{fruit_class}_{is_defective}_{rounded_bbox}"
            signatures.append(signature)
        
        # Sort signatures to ensure consistent ordering regardless of detection order
        signatures.sort()
        return "_".join(signatures)
    
    def _capture_photo_internal(self, frame, detection_results):
        """
        Internal method to capture and save a photo of the detected fruits.
        
        Args:
            frame: The camera frame to save
            detection_results: List of detection results
            
        Returns:
            str: Path to the saved image, or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename based on detection content
            if len(detection_results) == 1:
                result = detection_results[0]
                fruit_cls = result['fruit_class']
                is_def = result['is_defective']
                full_filename = f"{fruit_cls}_{'defective' if is_def else 'normal'}_full_{timestamp}.{self.image_format}"
            else:
                full_filename = f"multi_fruit_all_detections_{timestamp}.{self.image_format}"
                
            full_filepath = Path(self.image_save_path) / full_filename
            
            # Create a copy of the frame with bounding boxes drawn
            full_frame = frame.copy()
            for result in detection_results:
                bbox = result['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0) if not result['is_defective'] else (0, 0, 255)  # Green for normal, red for defective
                cv2.rectangle(full_frame, (x1, y1), (x2, y2), color, 2)
            
            # Save the captured image
            success = cv2.imwrite(str(full_filepath), full_frame)
            if success:
                self.logger.info(f"Delayed photo captured: {full_filepath}")
                return str(full_filepath)
            else:
                self.logger.error(f"Failed to save delayed photo: {full_filepath}")
                return None
        except Exception as e:
            self.logger.error(f"Error capturing delayed photo: {e}")
            return None
    
    def process_frame(self, frame, detection_results):
        """
        Process a frame and determine if a photo should be captured based on consistent detection.
        
        Args:
            frame: Current camera frame
            detection_results: List of detection results from the detection pipeline
            
        Returns:
            str or None: Path to captured image if captured, None otherwise
        """
        with self.lock:
            current_signature = self._get_detection_signature(detection_results)
            current_time = time.time()
            
            # If no detections, cancel any pending capture
            if not detection_results:
                if self.pending_capture:
                    self.pending_capture.cancel()
                    self.pending_capture = None
                    self.logger.debug("Cancelled pending capture due to no detections")
                
                self.current_detection_signature = None
                self.detection_start_time = None
                return None
            
            # If detection signature has changed, reset the timer
            if current_signature != self.current_detection_signature:
                if self.pending_capture:
                    self.pending_capture.cancel()
                    self.logger.debug("Cancelled previous pending capture due to detection change")
                
                # Start new timer for this detection
                self.current_detection_signature = current_signature
                self.detection_start_time = current_time
                self.pending_capture = threading.Timer(
                    self.delay_duration,
                    self._capture_photo_internal,
                    args=(frame, detection_results)
                )
                self.pending_capture.start()
                
                self.logger.debug(f"Started new capture timer for detection: {current_signature}")
            else:
                # Same detection - check if timer has already elapsed
                elapsed_time = current_time - self.detection_start_time
                if elapsed_time >= self.delay_duration and not self.pending_capture.is_alive():
                    # Timer already finished and photo was captured
                    # This is just for logging purposes
                    self.logger.debug(f"Detection maintained for {elapsed_time:.2f}s, photo already captured")
            
            # Return None since capture happens after delay (or was already handled)
            return None
    
    def cleanup(self):
        """Clean up resources and cancel any pending captures."""
        with self.lock:
            if self.pending_capture:
                self.pending_capture.cancel()
                self.logger.info("Cancelled pending capture during cleanup")
        
        self.logger.info("Delayed camera capturer cleaned up")