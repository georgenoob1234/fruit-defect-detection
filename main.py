#!/usr/bin/env python3
"""
Main application for the Fruit Defect Detection System with Camera Integration.

This application runs a continuous loop that:
- Captures frames from a camera
- Detects fruits using the fruit detection model
- Classifies defects using the defect detection model
- Implements debouncing to avoid repeated notifications
- Captures photos of defective fruits
- Sends detection data to a web API
- Sends notifications via Telegram bot
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logging_utils import setup_logging
from src.config_loader import load_config
from src.camera.camera_handler import CameraHandler
from src.detection.fruit_detector import FruitDetector
from src.detection.defect_detector import DefectSegmenter
from src.api.api_handler import APIHandler
from src.detection.detection_pipeline import DetectionPipeline
from src.gui.gui_handler import GUIHandler


class FruitDefectDetectionApp:
    def __init__(self):
        # Load configurations
        self.model_config = load_config('config/model_config.yaml')
        self.app_config = load_config('config/app_config.yaml')
        self.telegram_config = load_config('config/telegram_config.yaml')
        self.telegram_users = load_config('config/telegram_users.yaml')
        
        # Setup logging
        setup_logging(
            level=self.app_config['logging']['level'],
            log_file=self.app_config['logging']['file_path']
        )
        self.logger = logging.getLogger(__name__)
        
        # Check if static image mode is enabled
        self.use_static_image = self.app_config['static_image']['enabled']
        self.static_image_path = self.app_config['static_image']['image_path']
        
        if self.use_static_image:
            # Use static image instead of camera
            self.logger.info(f"Static image mode enabled: {self.static_image_path}")
        else:
            # Initialize camera as before
            self.camera = CameraHandler(
                source=self.app_config['camera']['source'],
                width=self.app_config['camera']['width'],
                height=self.app_config['camera']['height'],
                fps=self.app_config['camera']['fps']
            )
        
        self.fruit_detector = FruitDetector(
            model_path=self.model_config['fruit_detection']['model_path'],
            confidence_threshold=self.model_config['fruit_detection']['confidence_threshold'],
            target_classes=self.model_config['fruit_detection']['target_classes']
        )
        
        # Check if defect detection is enabled
        self.defect_detection_enabled = self.model_config['defect_detection'].get('enabled', True)
        
        if self.defect_detection_enabled:
            self.defect_segmenter = DefectSegmenter(
                model_path=self.model_config['defect_detection']['model_path'],
                confidence_threshold=self.model_config['defect_detection']['confidence_threshold'],
                target_classes=self.model_config['defect_detection']['target_classes']
            )
        else:
            self.defect_segmenter = None
            self.logger.info("Defect detection is disabled in configuration")
        
        self.api_handler = APIHandler(
            base_url=self.app_config['api']['base_url'],
            endpoint=self.app_config['api']['endpoint'],
            timeout=self.app_config['api']['timeout']
        )
        
        self.detection_pipeline = DetectionPipeline(
            self.fruit_detector,
            self.defect_segmenter,
            self.api_handler
        )
        
        
        # Photo capture settings
        self.capture_defective_only = self.app_config['photo_capture']['capture_defective_only']
        self.image_save_path = self.app_config['photo_capture']['image_save_path']
        self.image_format = self.app_config['photo_capture']['image_format']
        
        # GUI settings
        self.show_gui = self.app_config['gui']['show_gui']
        self.show_defect_status = self.app_config['gui']['show_defect_status']
        
        # Initialize GUI handler if GUI is enabled
        if self.show_gui:
            self.gui_handler = GUIHandler(show_defect_status=self.show_defect_status)
        
        # API settings
        self.api_enabled = self.app_config['api']['enabled']
        
        # Telegram settings
        self.telegram_enabled = self.telegram_config['telegram'].get('enable_telegram', True)
        self.telegram_token = self.telegram_config['telegram']['bot_token']
        self.telegram_user_ids = self.telegram_users['telegram_users']['user_ids']
        
        # Create save directory if it doesn't exist
        Path(self.image_save_path).mkdir(parents=True, exist_ok=True)
        
        # Deduplication tracking - tracks the last sent detection for each fruit class
        self.last_sent_detection = {}  # key: fruit_class, value: dict with is_defective and other properties (excluding confidence)
        
        # Exit flag for graceful shutdown
        self.exit_flag = threading.Event()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)


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

    def _signal_handler(self, signum, frame):
        """Handle termination signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.exit_flag.set()


    def capture_photo(self, frame, fruit_class, is_defective, detection_data):
       """
       Capture and save photos of detected fruits.
       Only save full frame images with bounding boxes, no cropping.
       """
       try:
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           
           # Get all detections from detection_data (could be single detection or list)
           all_detections = []
           if isinstance(detection_data, list):
               # Multiple detections case
               all_detections = detection_data
           else:
               # Single detection case - convert to list
               all_detections = [detection_data]
           
           saved_paths = []
           
           # Save full frame image with all bounding boxes
           full_frame = frame.copy()
           for det in all_detections:
               bbox = det['bbox']
               x1, y1, x2, y2 = map(int, bbox)
               color = (0, 255, 0) if not det['is_defective'] else (0, 0, 255)  # Green for normal, red for defective
               cv2.rectangle(full_frame, (x1, y1), (x2, y2), color, 2)
           
           # Create filename based on detection content
           if len(all_detections) == 1:
               det = all_detections[0]
               fruit_cls = det['fruit_class']
               is_def = det['is_defective']
               full_filename = f"{fruit_cls}_{'defective' if is_def else 'normal'}_full_{timestamp}.{self.image_format}"
           else:
               full_filename = f"multi_fruit_all_detections_{timestamp}.{self.image_format}"
               
           full_filepath = os.path.join(self.image_save_path, full_filename)
           
           cv2.imwrite(full_filepath, full_frame)
           self.logger.info(f"Full frame photo with bounding boxes captured: {full_filepath}")
           saved_paths.append(full_filepath)
           
           # Return the primary image path for compatibility
           return saved_paths[0] if saved_paths else None
           
       except Exception as e:
           self.logger.error(f"Error capturing photo: {e}")
           return None

    def run(self):
        """Main application loop."""
        self.logger.info("Starting Fruit Defect Detection Application...")
        
        if not self.use_static_image:
            if not self.camera.start():
                self.logger.error("Failed to start camera")
                return
            
            self.logger.info("Camera started successfully")
        
        try:
            while not self.exit_flag.is_set():
                if self.use_static_image:
                    # Read static image
                    self.logger.info(f"Attempting to read static image from: {self.static_image_path}")
                    frame = cv2.imread(self.static_image_path)
                    if frame is None:
                        self.logger.error(f"Failed to read static image: {self.static_image_path}")
                        # Instead of breaking, wait a bit and continue the loop
                        time.sleep(0.1)
                        continue
                    self.logger.info(f"Successfully read static image with shape: {frame.shape}")
                    ret = True
                else:
                    # Read frame from camera
                    ret, frame = self.camera.read()
                
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                try:
                    # Run detection pipeline
                    detection_results, processed_frame = self.detection_pipeline.process_frame(frame)
                    
                    # Copy frame for display to avoid modifying the original during processing
                    display_frame = processed_frame.copy() if self.show_gui else None
                    
                    # Prepare detection data for all results
                    all_detection_data = []
                    for result in detection_results:
                        fruit_class = result['fruit_class']
                        is_defective = result['is_defective']
                        confidence = result['confidence']
                        bbox = result['bbox']
                        
                        # Prepare detection data
                        detection_data = {
                            'fruit_class': fruit_class,
                            'is_defective': is_defective,
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat(),
                            'bbox': bbox
                        }
                        all_detection_data.append(detection_data)
                    
                    # Only capture photo if there are detections
                    # Pass all detection data to capture_photo method
                    if detection_results:
                        image_path = self.capture_photo(frame, None, None, all_detection_data)
                    else:
                        image_path = None
                        self.logger.info("No detections found, skipping photo capture")
                    
                    # Process detection results
                    for i, result in enumerate(detection_results):
                        fruit_class = result['fruit_class']
                        is_defective = result['is_defective']
                        confidence = result['confidence']
                        bbox = result['bbox']
                        
                        # Get the corresponding detection data
                        detection_data = all_detection_data[i]
                        detection_data['image_path'] = image_path
                        
                        # Send to API if enabled
                        if self.api_enabled:
                            try:
                                # Only send image to API if the fruit is defective
                                api_image_path = image_path if is_defective else None
                                self.api_handler.send_detection(detection_data, api_image_path)
                                self.logger.info(f"Detection data sent to API: {detection_data}")
                            except Exception as e:
                                self.logger.error(f"Error sending detection to API: {e}")
                        
                        # Send to Telegram if enabled
                        if self.telegram_enabled:
                            try:
                                self._send_telegram_notification(detection_data, image_path)
                            except Exception as e:
                                self.logger.error(f"Error sending Telegram notification: {e}")
                        
                        self.logger.info(f"Processed detection: {fruit_class}, defective: {is_defective}, confidence: {confidence:.2f}")
                except Exception as e:
                    self.logger.error(f"Error during detection processing: {e}")
                    # Continue the loop even if there's an error in detection processing
                    continue
               
                # Log the detection for Telegram regardless of whether notifications were sent
                # Only log if there were actual detections
                if detection_results:
                    try:
                        # Log the detection for potential retrieval via /showlogs command
                        if self.telegram_enabled:
                            # Import telegram bot module if available
                            from src.telegram.telegram_bot import TelegramBot
                            bot = TelegramBot(self.telegram_token, users_config_path='config/telegram_users.yaml')
                            # Log each detection
                            for detection_data in all_detection_data:
                                bot.log_detection(detection_data)
                        # Log the first detection details
                        first_detection = all_detection_data[0] if all_detection_data else None
                        if first_detection:
                            self.logger.info(f"Logged detection: {first_detection['fruit_class']}, defective: {first_detection['is_defective']}, confidence: {first_detection['confidence']:.2f}")
                    except Exception as e:
                        self.logger.error(f"Error logging detection: {e}")
                else:
                    # No detections case
                    self.logger.info("No detections to log")
               
                # Display frame if GUI is enabled
                if self.show_gui and display_frame is not None:
                    # Use GUI handler to draw detection results and show frame
                    continue_running = self.gui_handler.show_frame(display_frame, detection_results)
                    if not continue_running:
                        self.logger.info("Exit key pressed, stopping application...")
                        break
                
                # If using static image, display the results and keep the application running
                if self.use_static_image:
                    self.logger.info("Static image processed, keeping application running for viewing results")
                    # For static image mode, keep the display running until user presses 'q'
                    if self.show_gui:
                        # Keep the application running to view the results
                        self.logger.info("Press 'q' in the GUI window to exit the application")
                        while not self.exit_flag.is_set():
                            # Continue showing the processed frame with detections
                            continue_running = self.gui_handler.show_frame(processed_frame, detection_results)
                            if not continue_running:
                                self.logger.info("Exit key pressed, stopping application...")
                                break
                            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                    else:
                        # If GUI is not enabled, wait for exit flag or keyboard interrupt
                        self.logger.info("Static image processed. Press Ctrl+C to exit the application")
                        while not self.exit_flag.is_set():
                            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    # Instead of breaking, continue the main loop to allow for continuous operation
                    continue # Continue the main loop instead of breaking
            
            self.logger.info("Application loop ended")
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, stopping application...")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def _send_telegram_notification(self, detection_data, image_path=None):
        """Send detection notification via Telegram."""
        # Import telegram bot module if available
        try:
            from src.telegram.telegram_bot import TelegramBot
            
            # Check if this detection is a duplicate that should be suppressed
            if not self._is_duplicate_detection(detection_data):
                # Initialize the bot
                bot = TelegramBot(self.telegram_token, users_config_path='config/telegram_users.yaml')
                
                # Log the detection for potential retrieval via /showlogs command
                bot.log_detection(detection_data)
                
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
                        bot.send_message_async(user_id, message, image_path, timeout=30)
                        self.logger.info(f"Started async Telegram notification to user {user_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to start async Telegram message to {user_id}: {e}")
                
                # Update the tracking for the last sent detection
                self._update_last_sent_detection(detection_data)
            else:
                self.logger.info(f"Suppressed duplicate notification for {detection_data['fruit_class']} (defective: {detection_data['is_defective']})")
        except ImportError:
            self.logger.warning("Telegram bot module not found, skipping notifications")
        except Exception as e:
            self.logger.error(f"Error sending Telegram notification: {e}")

    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        
        if self.show_gui:
            self.gui_handler.cleanup()
        
        if not self.use_static_image:
            self.camera.stop()
        
        self.logger.info("Application shutdown complete")


def main():
    app = FruitDefectDetectionApp()
    app.run()


if __name__ == "__main__":
    main()