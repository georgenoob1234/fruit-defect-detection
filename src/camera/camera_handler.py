"""
Camera Handler module for the Fruit Defect Detection System.

This module provides a CameraHandler class to manage camera operations
including initialization, frame capture, and cleanup.
"""
import cv2
import logging
import threading
import time


class CameraHandler:
    """
    A class to handle camera operations including initialization, 
    frame capture, and cleanup.
    """
    
    def __init__(self, source=0, width=640, height=480, fps=30):
        """
        Initialize the camera handler with specified parameters.
        
        Args:
            source: Camera source (0 for default camera, or path to video file)
            width: Camera frame width in pixels
            height: Camera frame height in pixels
            fps: Camera frames per second
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"CameraHandler initialized with source={source}, "
                        f"resolution={width}x{height}, fps={fps}")
    
    def start(self):
        """
        Start the camera capture.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera with source: {self.source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify the properties were set
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera started: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
            return False
    
    def read(self):
        """
        Read a frame from the camera.
        
        Returns:
            tuple: (bool, frame) where bool indicates success and frame is the image array
        """
        if self.cap is None or not self.cap.isOpened():
            self.logger.warning("Camera not started or already released")
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Failed to read frame from camera")
                return False, None
            
            return True, frame
            
        except Exception as e:
            self.logger.error(f"Error reading frame from camera: {e}")
            return False, None
    
    def stop(self):
        """
        Stop and release the camera resources.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.logger.info("Camera released")
        
        # Close all OpenCV windows if any are open
        cv2.destroyAllWindows()