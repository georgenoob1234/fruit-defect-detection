"""
GUI Handler module for the Fruit Defect Detection System.

This module provides a GUIHandler class to manage the graphical user interface,
displaying the camera feed with bounding boxes around detected objects,
class names, and defect status.
"""
import cv2
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional


class GUIHandler:
    """
    A class to handle the graphical user interface for the fruit defect detection system.
    Displays camera feed with bounding boxes, class names, and defect status.
    """
    
    def __init__(self, window_name: str = "Fruit Defect Detection", show_defect_status: bool = True):
        """
        Initialize the GUI handler.
        
        Args:
            window_name: Name of the display window
            show_defect_status: When True, shows both fruit type and defect status;
                               when False, shows only fruit type without defect indication
        """
        self.window_name = window_name
        self.show_defect_status = show_defect_status
        self.logger = logging.getLogger(__name__)
        
        # Initialize the display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        self.logger.info(f"GUI Handler initialized with window name: {window_name}, show_defect_status: {show_defect_status}")
    
    def draw_detections(self, frame: np.ndarray, detection_results: List[Dict]) -> np.ndarray:
        """
        Draw detection results on the frame.
        
        Args:
            frame: Input frame to draw on
            detection_results: List of detection results, each containing:
                - fruit_class (str): The type of fruit detected
                - is_defective (bool): Whether the fruit is defective
                - confidence (float): Confidence score (0.0-1.0)
                - bbox (list): Bounding box coordinates [x1, y1, x2, y2]
                - masks (list, optional): List of segmentation masks for defects
        
        Returns:
            np.ndarray: Frame with drawn detection results
        """
        # Make a copy of the frame to avoid modifying the original
        display_frame = frame.copy()
        
        for result in detection_results:
            bbox = result['bbox']
            fruit_class = result['fruit_class']
            is_defective = result['is_defective']
            confidence = result['confidence']
            masks = result.get('masks', [])
            
            # Determine colors based on defect status (only if showing defect status)
            if self.show_defect_status:
                box_color = (0, 0, 255) if is_defective else (0, 255, 0)  # Red for defective, green for normal
                text_color = (0, 0, 255) if is_defective else (0, 255, 0)
            else:
                # When not showing defect status, use a consistent color (green) for all fruits
                box_color = (0, 255, 0) # Always green when defect status is hidden
                text_color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
            
            # Draw segmentation masks if available and fruit is defective
            if is_defective and masks:
                for mask in masks:
                    # Draw the mask on the display frame (this is a simplified approach)
                    # In a real implementation, we would properly overlay the segmentation mask
                    mask_visual = np.zeros_like(display_frame)
                    mask_visual[:, :, 0] = mask * 50  # Add red tint to defective areas
                    
                    # Apply the mask as a transparent overlay
                    display_frame = cv2.addWeighted(display_frame, 1.0, mask_visual, 0.3, 0)
            
            # Prepare label text based on toggle setting
            if self.show_defect_status:
                status_text = "DEFECTIVE" if is_defective else "NORMAL"
                label = f"{fruit_class.upper()} - {status_text} ({confidence:.2f})"
            else:
                # When not showing defect status, only show fruit class and confidence
                label = f"{fruit_class.upper()} ({confidence:.2f})"
            
            # Calculate text size to create background rectangle
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_bg_corner1 = (bbox[0], bbox[1] - text_size[1] - 10)
            text_bg_corner2 = (bbox[0] + text_size[0], bbox[1])
            
            # Draw background rectangle for text with transparency effect
            cv2.rectangle(display_frame, text_bg_corner1, text_bg_corner2, box_color, -1)
            cv2.rectangle(display_frame, text_bg_corner1, text_bg_corner2, (255, 255, 255), 1)
            
            # Draw label text in white
            cv2.putText(display_frame, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add a confidence bar below the bounding box (only if showing defect status)
            if self.show_defect_status:
                bar_width = int((confidence) * (bbox[2] - bbox[0]))
                cv2.rectangle(display_frame, (bbox[0], bbox[3] + 5),
                             (bbox[0] + bar_width, bbox[3] + 15),
                             (0, 255, 0) if not is_defective else (0, 0, 25), -1)
                cv2.rectangle(display_frame, (bbox[0], bbox[3] + 5),
                             (bbox[2], bbox[3] + 15), (200, 200, 200), 1)
        
        # Add overall information panel
        height, width = display_frame.shape[:2]
        cv2.rectangle(display_frame, (10, 10), (300, 110), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (300, 110), (200, 200, 200), 2)
        
        # Add text to the information panel
        cv2.putText(display_frame, "FRUIT DEFECT DETECTION", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Detections: {len(detection_results)}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Only show defective count if showing defect status
        if self.show_defect_status:
            defective_count = sum(1 for r in detection_results if r['is_defective'])
            cv2.putText(display_frame, f"Defective: {defective_count}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if defective_count > 0 else (0, 255, 0), 1)
            cv2.putText(display_frame, "Press 'Q' to quit", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(display_frame, "Defect Status: HIDDEN", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display_frame, "Press 'Q' to quit", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return display_frame
    
    def show_frame(self, frame: np.ndarray, detection_results: List[Dict] = None) -> bool:
        """
        Display the frame with detection results.
        
        Args:
            frame: Frame to display
            detection_results: Optional detection results to draw on the frame
        
        Returns:
            bool: True if continue running (user didn't press 'q'), False otherwise
        """
        if detection_results:
            display_frame = self.draw_detections(frame, detection_results)
        else:
            display_frame = frame.copy()
        
        # Show the frame
        cv2.imshow(self.window_name, display_frame)
        
        # Check for exit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        
        return True
    
    def cleanup(self):
        """
        Clean up GUI resources.
        """
        cv2.destroyAllWindows()
        self.logger.info("GUI resources cleaned up")
    
    def set_window_size(self, width: int, height: int):
        """
        Set the window size.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        cv2.resizeWindow(self.window_name, width, height)