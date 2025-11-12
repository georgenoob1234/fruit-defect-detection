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
from src.utils.ood_detector import OODDetector

from src.utils.background_image_monitor import BackgroundImageMonitor


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
        self.defect_detection_enabled = self.model_config['defect_segmentation'].get('enabled', True)
        
        if self.defect_detection_enabled:
            self.defect_segmenter = DefectSegmenter(
                model_path=self.model_config['defect_segmentation']['model_path'],
                confidence_threshold=self.model_config['defect_segmentation']['confidence_threshold'],
                target_classes=self.model_config['defect_segmentation']['target_classes']
            )
        else:
            self.defect_segmenter = None
            self.logger.info("Defect detection is disabled in configuration")
        
        self.api_handler = APIHandler(
            base_url=self.app_config['api']['base_url'],
            endpoint=self.app_config['api']['endpoint'],
            timeout=self.app_config['api']['timeout']
        )
        
        # Initialize OOD detector with configuration
        ood_config = self.model_config.get('ood_detection', {
            'enabled': True,
            'ood_threshold': 0.3,
            'confidence_threshold': 0.5,
            'max_class_variance': 0.7,
            'min_detection_ratio': 0.1,
            'known_classes': self.model_config['fruit_detection']['target_classes']
        })
        
        self.ood_detector = OODDetector(ood_config)
        
        self.detection_pipeline = DetectionPipeline(
            self.fruit_detector,
            self.defect_segmenter,
            self.api_handler,
            self.ood_detector
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
        
        # Initialize Telegram bot instance once if enabled
        if self.telegram_enabled and self.telegram_token and self.telegram_token != "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            try:
                from src.telegram.telegram_bot import TelegramBot
                self.telegram_bot = TelegramBot(self.telegram_token, users_config_path='config/telegram_users.yaml')
                self.logger.info("Telegram bot initialized successfully")
            except ImportError:
                self.logger.warning("Telegram bot module not found, skipping notifications")
                self.telegram_bot = None
            except Exception as e:
                self.logger.error(f"Error initializing Telegram bot: {e}")
                self.telegram_bot = None
        else:
            self.telegram_bot = None
            if self.telegram_enabled:
                self.logger.warning("Telegram is enabled but no valid token provided, skipping notifications")
        
        # Create save directory if it doesn't exist
        Path(self.image_save_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize background image monitor to run as a separate thread
        self.image_monitor = BackgroundImageMonitor(
            config_path='config/app_config.yaml',
            folder_path=self.image_save_path
        )
        
        # Start the image monitor in a separate thread
        self.image_monitor.start_monitoring()
        
        
        # Exit flag for graceful shutdown
        self.exit_flag = threading.Event()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)



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
                        
                        # Prepare internal detection data (with bbox for GUI)
                        internal_detection_data = {
                            'fruit_class': fruit_class,
                            'is_defective': is_defective,
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat(),
                            'bbox': bbox
                        }
                        
                        # Prepare API detection data (without bbox, with image_path)
                        api_detection_data = {
                            'fruit_class': fruit_class,
                            'is_defective': is_defective,
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat(),
                            'image_path': None  # Will be set after photo capture
                        }
                        
                        all_detection_data.append(internal_detection_data)
                        # Store API version separately or modify the same data after photo capture
                        # For now, we'll modify the same data after photo capture
                    
                    # Only capture photo if there are detections and settings allow it
                    # Pass all detection data to capture_photo method
                    if detection_results:
                        # Check if we should only capture photos when defects are detected
                        should_capture = True
                        if self.capture_defective_only:
                            # Only capture if at least one detection is defective
                            should_capture = any(detection['is_defective'] for detection in all_detection_data)
                        
                        if should_capture:
                            image_path = self.capture_photo(frame, None, None, all_detection_data)
                            self.logger.info(f"Photo captured based on capture_defective_only setting: {self.capture_defective_only}")
                        else:
                            image_path = None
                            self.logger.info("No defective detections found, skipping photo capture due to capture_defective_only setting")
                    else:
                        image_path = None
                        self.logger.info("No detections found, skipping photo capture")
                    
                    # Process detection results for API - consolidate to single call
                    if self.api_enabled and detection_results:
                        try:
                            # Prepare consolidated API detection data
                            # Only send image to API if at least one fruit is defective
                            at_least_one_defective = any(detection['is_defective'] for detection in all_detection_data)
                            api_image_path = image_path if at_least_one_defective else None
                            
                            # Send consolidated detection data to API
                            self._send_consolidated_api_detection(all_detection_data, api_image_path)
                            self.logger.info(f"Consolidated detection data sent to API for {len(all_detection_data)} fruits")
                        except Exception as e:
                            self.logger.error(f"Error sending consolidated detection to API: {e}")
                    
                    # Send single consolidated Telegram notification for all detections in the frame
                    if self.telegram_enabled and detection_results:
                        try:
                            self._send_consolidated_telegram_notification(all_detection_data, image_path)
                        except Exception as e:
                            self.logger.error(f"Error sending consolidated Telegram notification: {e}")
                    
                    # Log individual detections for processing reference
                    for i, result in enumerate(detection_results):
                        fruit_class = result['fruit_class']
                        is_defective = result['is_defective']
                        confidence = result['confidence']
                        
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
                        if self.telegram_enabled and self.telegram_bot:
                            # Use the existing telegram bot instance to log each detection
                            for detection_data in all_detection_data:
                                self.telegram_bot.log_detection(detection_data)
                        # Log the consolidated detection details
                        if all_detection_data:
                            fruit_names = list(set([det['fruit_class'] for det in all_detection_data]))  # Get unique fruit names
                            self.logger.info(f"Logged detection: {', '.join(fruit_names)}, Total fruits detected: {len(all_detection_data)}")
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
        # Check if telegram is enabled and we have a bot instance
        if not self.telegram_enabled or not self.telegram_bot:
            return
            
        try:
            # Log the detection for potential retrieval via /showlogs command
            self.telegram_bot.log_detection(detection_data)
            
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
                    
        except Exception as e:
            self.logger.error(f"Error sending Telegram notification: {e}")

    def _send_consolidated_telegram_notification(self, all_detection_data, image_path=None):
        """Send a consolidated detection notification via Telegram for all fruits in the frame."""
        # Check if telegram is enabled and we have a bot instance
        if not self.telegram_enabled or not self.telegram_bot:
            return
            
        try:
            # Log all detections for potential retrieval via /showlogs command
            for detection_data in all_detection_data:
                self.telegram_bot.log_detection(detection_data)
            
            # Consolidate fruit names
            fruit_names = []
            defective_status = "No"  # Default to No unless at least one is defective
            avg_confidence = 0.0
            
            for detection in all_detection_data:
                fruit_class = detection['fruit_class']
                if fruit_class not in fruit_names:
                    fruit_names.append(fruit_class)
                if detection['is_defective']:
                    defective_status = "Yes"
                avg_confidence += detection['confidence']
            
            avg_confidence = avg_confidence / len(all_detection_data) if all_detection_data else 0.0
            
            # Create consolidated message
            fruit_list = ', '.join(fruit_names)
            message = (
                f"🍎 Fruit Detection Alert 🍎\n"
                f"Fruits: {fruit_list}\n"
                f"Defective: {defective_status}\n"
                f"Confidence: {avg_confidence:.2f}\n"
                f"Time: {datetime.now().isoformat()}"
            )
            
            # Send message to all authorized users asynchronously to avoid blocking
            for user_id in self.telegram_user_ids:
                try:
                    # Use async method to avoid blocking the main thread
                    self.telegram_bot.send_message_async(user_id, message, image_path, timeout=30)
                    self.logger.info(f"Started async consolidated Telegram notification to user {user_id} for fruits: {fruit_list}")
                except Exception as e:
                    self.logger.error(f"Failed to start async Telegram message to {user_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error sending consolidated Telegram notification: {e}")

    def _send_consolidated_api_detection(self, all_detection_data, image_path=None):
        """Send consolidated detection data to the API for all fruits in the frame."""
        if not all_detection_data:
            return
            
        # Consolidate detection data - create a list of all detections in this frame
        consolidated_data = {
            'detections': [],  # List of individual detection objects
            'frame_info': {
                'total_detections': len(all_detection_data),
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }
        }
        
        # Add each detection to the consolidated data
        for detection in all_detection_data:
            detection_info = {
                'fruit_class': detection['fruit_class'],
                'is_defective': detection['is_defective'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']  # Include bounding box info if needed by API
            }
            consolidated_data['detections'].append(detection_info)
        
        # Determine if any detection is defective to decide if we send the image
        at_least_one_defective = any(detection['is_defective'] for detection in all_detection_data)
        api_image_path = image_path if at_least_one_defective else None
        
        try:
            # Send the consolidated data to the API
            success = self.api_handler.send_detection(consolidated_data, api_image_path)
            if success:
                self.logger.info(f"Successfully sent consolidated detection data to API: {len(all_detection_data)} fruits detected")
            else:
                self.logger.error(f"Failed to send consolidated detection data to API")
        except Exception as e:
            self.logger.error(f"Error sending consolidated detection data to API: {e}")

    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        
        if self.show_gui:
            self.gui_handler.cleanup()
        
        if not self.use_static_image:
            self.camera.stop()
        
        # Stop image monitor if it's running
        try:
            self.image_monitor.stop_monitoring()
        except:
            # The monitor might not be running in continuous mode, which is fine
            pass
        
        # Clean up telegram bot if it exists
        if hasattr(self, 'telegram_bot') and self.telegram_bot:
            try:
                self.telegram_bot.cleanup()
                self.logger.info("Telegram bot cleaned up successfully")
            except Exception as e:
                self.logger.error(f"Error cleaning up telegram bot: {e}")
        
        self.logger.info("Application shutdown complete")


def main():
    app = FruitDefectDetectionApp()
    app.run()


if __name__ == "__main__":
    main()