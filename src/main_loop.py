"""
Main loop implementation for the Fruit Defect Detection System.

This module provides the core processing loop that coordinates
between the camera, detectors, API handler, and Telegram bot.
"""

import time
import threading
from datetime import datetime
from pathlib import Path
import cv2
import logging

from src.detection.detection_pipeline import DetectionPipeline
from src.gui.gui_handler import GUIHandler
from src.utils.detection_status_logger import DetectionStatusLogger, get_detection_status_logger


class MainLoop:
    """
    Main application loop that coordinates all components of the fruit defect detection system.
    """
    
    def __init__(self, config, fruit_detector, defect_detector, api_handler, camera_handler, telegram_bot=None):
        """
        Initialize the main loop with required components.
        
        Args:
            config: Application configuration dictionary
            fruit_detector: Instance of FruitDetector
            defect_detector: Instance of DefectDetector (can be None if defect detection is disabled)
            api_handler: Instance of APIHandler
            camera_handler: Instance of CameraHandler
            telegram_bot: Optional TelegramBot instance
        """
        self.config = config
        self.fruit_detector = fruit_detector
        self.defect_detector = defect_detector
        self.api_handler = api_handler
        self.camera_handler = camera_handler
        self.telegram_bot = telegram_bot
        
        # Check if defect detection is enabled based on whether defect_detector is provided
        self.defect_detection_enabled = defect_detector is not None
        
        # Get telegram enabled status from config
        # Check for the telegram settings in the main config
        telegram_settings = config.get('telegram', {})
        if isinstance(telegram_settings, dict):
            self.telegram_enabled = telegram_settings.get('enable_telegram', True)
        else:
            # Fallback to True if telegram settings are not properly configured
            self.telegram_enabled = True
        
        # Extract configuration values
        self.show_gui = config['gui']['show_gui']
        self.show_defect_status = config['gui']['show_defect_status']
        self.api_enabled = config['api']['enabled']
        self.debounce_time = config['debouncing']['detection_debounce_time']
        self.capture_defective_only = config['photo_capture']['capture_defective_only']
        self.image_save_path = config['photo_capture']['image_save_path']
        self.image_format = config['photo_capture']['image_format']
        self.telegram_user_ids = config.get('telegram_user_ids', [])
        
        # Debouncing tracking
        self.last_detection_times = {}
        
        # Deduplication tracking - tracks the last sent detection for each fruit class
        self.last_sent_detection = {}  # key: fruit_class, value: dict with is_defective and other properties (excluding confidence)
        
        # Detection status tracking
        self.current_detection_count = 0
        self.was_detecting = False
        
        # Exit flag for graceful shutdown
        self.exit_flag = threading.Event()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize GUI handler if GUI is enabled
        if self.show_gui:
            self.gui_handler = GUIHandler(show_defect_status=self.show_defect_status)
        
        # Create detection pipeline
        self.detection_pipeline = DetectionPipeline(fruit_detector, defect_detector, api_handler)
        
        # Initialize detection status logger
        self.detection_status_logger = get_detection_status_logger()
        
        # Create save directory if it doesn't exist
        Path(self.image_save_path).mkdir(parents=True, exist_ok=True)

    def should_process_detection(self, fruit_class):
        """
        Check if enough time has passed since the last detection of this fruit class.
        
        Args:
            fruit_class (str): The class of fruit detected
            
        Returns:
            bool: True if detection should be processed, False otherwise
        """
        current_time = time.time()
        
        if fruit_class not in self.last_detection_times:
            # First detection of this class
            self.last_detection_times[fruit_class] = current_time
            return True
        
        # Check if debounce time has passed
        if current_time - self.last_detection_times[fruit_class] >= self.debounce_time:
            self.last_detection_times[fruit_class] = current_time
            return True
        
        return False

    def _is_duplicate_detection(self, detection_data):
        """
        Check if this detection is a duplicate that should be suppressed.
        Only suppress notifications if it's the same fruit type with same defect status and properties
        (excluding confidence values which naturally vary).
        
        Args:
            detection_data (dict): Detection information to check
            
        Returns:
            bool: True if this is a duplicate that should be suppressed, False otherwise
        """
        fruit_class = detection_data['fruit_class']
        is_defective = detection_data['is_defective']
        
        # Create a key for comparison that excludes confidence (which naturally varies)
        detection_key = (
            detection_data['fruit_class'],
            detection_data['is_defective']
        )
        
        # Check if we have sent a notification for this fruit class before
        if fruit_class in self.last_sent_detection:
            # Compare with the last sent detection for this fruit class
            last_detection = self.last_sent_detection[fruit_class]
            
            # Create a key for the last sent detection
            last_detection_key = (
                last_detection['fruit_class'],
                last_detection['is_defective']
            )
            
            # If keys match, this is a duplicate
            if detection_key == last_detection_key:
                return True
        
        # This is a new detection or different from the last one sent
        return False


    def _update_last_sent_detection(self, detection_data):
        """
        Update the tracking for the last sent detection of this fruit class.
        
        Args:
            detection_data (dict): Detection information that was sent
        """
        fruit_class = detection_data['fruit_class']
        # Store the detection properties that define uniqueness (fruit class and defect status)
        self.last_sent_detection[fruit_class] = {
            'fruit_class': detection_data['fruit_class'],
            'is_defective': detection_data['is_defective']
        }


    def capture_photo(self, frame, fruit_class, is_defective, detection_data):
        """
        Capture and save a photo of the detected fruit.
        
        Args:
            frame: The camera frame to save
            fruit_class (str): The type of fruit detected
            is_defective (bool): Whether the fruit is defective
            detection_data (dict): Detection information
            
        Returns:
            str: Path to the saved image, or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{fruit_class}_{'defective' if is_defective else 'normal'}_{timestamp}.{self.image_format}"
            filepath = Path(self.image_save_path) / filename
            
            # Save the captured image
            success = cv2.imwrite(str(filepath), frame)
            if success:
                self.logger.info(f"Photo captured: {filepath}")
                return str(filepath)
            else:
                self.logger.error(f"Failed to save photo: {filepath}")
                return None
        except Exception as e:
            self.logger.error(f"Error capturing photo: {e}")
            return None

    def _send_telegram_notification(self, detection_data, image_path=None):
        """
        Send detection notification via Telegram.
        
        Args:
            detection_data (dict): Detection information
            image_path (str, optional): Path to the captured image
        """
        if not self.telegram_bot:
            return
            
        # Prepare message
        message = (
            f"🍎 Fruit Detection Alert 🍎\n"
            f"Fruit: {detection_data['fruit_class']}\n"
            f"Defective: {'Yes' if detection_data['is_defective'] else 'No'}\n"
            f"Confidence: {detection_data['confidence']:.2f}\n"
            f"Time: {detection_data['timestamp']}"
        )
        
        # Send message to all authorized users asynchronously to avoid blocking
        for user_id in self.telegram_user_ids:
            try:
                # Use async method to avoid blocking the main thread
                self.telegram_bot.send_message_async(user_id, message, image_path, timeout=30)
                self.logger.info(f"Started async Telegram notification to user {user_id}")
            except Exception as e:
                self.logger.error(f"Failed to start async Telegram message to {user_id}: {e}")

    def _prepare_detection_data(self, detection_result):
        """
        Prepare detection data with timestamp.
        
        Args:
            detection_result (dict): Raw detection result
            
        Returns:
            dict: Formatted detection data
        """
        return {
            'fruit_class': detection_result['fruit_class'],
            'is_defective': detection_result['is_defective'],
            'confidence': detection_result['confidence'],
            'timestamp': datetime.now().isoformat(),
            'bbox': detection_result['bbox']
        }

    def run(self):
        """
        Main application loop that runs continuously until stopped.
        """
        self.logger.info("Starting main application loop...")
        
        if not self.camera_handler.start():
            self.logger.error("Failed to start camera")
            return
        
        self.logger.info("Camera started successfully")
        
        try:
            while not self.exit_flag.is_set():
                # Read frame from camera
                ret, frame = self.camera_handler.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Copy frame for display to avoid modifying the original during processing
                display_frame = frame.copy() if self.show_gui else None
                
                try:
                    # Run detection pipeline
                    detection_results = self.detection_pipeline.process_frame(frame)
                    
                    # Update detection count and notify status logger if state changed
                    new_detection_count = len(detection_results)
                    if new_detection_count > 0 and self.current_detection_count == 0:
                        # Detection started (was 0, now > 0)
                        self.detection_status_logger.detection_started()
                        self.was_detecting = True
                    elif new_detection_count == 0 and self.current_detection_count > 0:
                        # Detection ended (was > 0, now 0)
                        self.detection_status_logger.detection_ended()
                        self.was_detecting = False
                    
                    self.current_detection_count = new_detection_count
                    
                    # Process each detection result
                    for result in detection_results:
                        fruit_class = result['fruit_class']
                        
                        # Prepare detection data
                        detection_data = self._prepare_detection_data(result)
                        
                        # Always capture photo for logging purposes regardless of debounce
                        image_path = self.capture_photo(frame, detection_data['fruit_class'],
                                                      detection_data['is_defective'], detection_data)
                        detection_data['image_path'] = image_path
                        
                        # Check debouncing for processing (API/Telegram notifications)
                        if self.should_process_detection(fruit_class):
                            # Send to API if enabled
                            if self.api_enabled:
                                try:
                                    # Only send image to API if the fruit is defective
                                    api_image_path = image_path if detection_data['is_defective'] else None
                                    self.api_handler.send_detection(detection_data, api_image_path)
                                    self.logger.info(f"Detection data sent to API: {detection_data}")
                                except Exception as e:
                                    self.logger.error(f"Error sending detection to API: {e}")
                            
                            # Send to Telegram if bot is available and enabled
                            if self.telegram_bot and self.telegram_enabled:
                                try:
                                    # Check if this detection is a duplicate that should be suppressed
                                    if not self._is_duplicate_detection(detection_data):
                                        # Log the detection with image path before sending notification
                                        self.telegram_bot.log_detection(detection_data)
                                        self._send_telegram_notification(detection_data, image_path)
                                        # Update the tracking for the last sent detection
                                        self._update_last_sent_detection(detection_data)
                                    else:
                                        self.logger.info(f"Suppressed duplicate notification for {detection_data['fruit_class']} (defective: {detection_data['is_defective']})")
                                except Exception as e:
                                    self.logger.error(f"Error sending Telegram notification: {e}")
                                    
                            self.logger.info(f"Processed detection: {fruit_class}, defective: {detection_data['is_defective']}, confidence: {detection_data['confidence']:.2f}")
                        else:
                            # Still log the detection but don't send notifications
                            if self.telegram_bot and self.telegram_enabled:
                                try:
                                    # Log the detection even if we don't send notifications
                                    self.telegram_bot.log_detection(detection_data)
                                    self.logger.info(f"Logged detection (debounced): {fruit_class}, defective: {detection_data['is_defective']}, confidence: {detection_data['confidence']:.2f}")
                                except Exception as e:
                                    self.logger.error(f"Error logging detection: {e}")
                except Exception as e:
                    self.logger.error(f"Error during detection processing: {e}")
                    # Continue the loop even if there's an error in detection processing
                    continue
                
                # Display frame if GUI is enabled
                if self.show_gui and display_frame is not None:
                    # Use GUI handler to draw detection results and show frame
                    continue_running = self.gui_handler.show_frame(display_frame, detection_results)
                    if not continue_running:
                        self.logger.info("Exit key pressed, stopping application...")
                        break
            
            self.logger.info("Main loop ended")
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, stopping application...")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        
        if self.show_gui:
            self.gui_handler.cleanup()
        
        # Clean up detection status logger if it exists
        if hasattr(self, 'detection_status_logger'):
            self.detection_status_logger.cleanup()
        
        self.camera_handler.stop()
        
        self.logger.info("Main loop cleanup complete")