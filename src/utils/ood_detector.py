"""
Out-of-Distribution (OOD) Detection module for the Fruit Defect Detection System.

This module implements various OOD detection techniques to identify when input images
contain objects or scenarios that fall outside the trained model's expected distribution,
thereby preventing false positives on unknown or unexpected inputs.
"""
import logging
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from typing import Tuple, List, Dict, Any


class OODDetector:
    """
    A class to perform Out-of-Distribution detection on input images.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the OOD detector with configuration parameters.
        
        Args:
            config: Configuration dictionary containing OOD detection parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # OOD detection parameters from config
        self.ood_threshold = self.config.get('ood_threshold', 0.3)
        self.enable_ood_detection = self.config.get('enabled', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.max_class_variance = self.config.get('max_class_variance', 0.7)
        self.min_detection_ratio = self.config.get('min_detection_ratio', 0.1)
        
        # Define known classes for fruit detection
        self.known_classes = self.config.get('known_classes', ['apple', 'banana', 'tomato'])
        
        self.logger.info(f"OOD detector initialized with threshold: {self.ood_threshold}")
        self.logger.info(f"Known classes: {self.known_classes}")
        
    def detect_ood(self, frame: np.ndarray, detections: List[Dict] = None) -> Tuple[bool, float, str]:
        """
        Detect if the input frame contains out-of-distribution content.
        
        Args:
            frame: Input frame to analyze
            detections: List of existing detections to analyze
            
        Returns:
            tuple: (is_ood: bool, confidence: float, reason: str)
        """
        if not self.enable_ood_detection:
            return False, 1.0, "OOD detection disabled"
        
        try:
            # Multiple OOD detection methods
            results = []
            
            # Method 1: Confidence-based OOD detection
            confidence_ood, conf_score, conf_reason = self._confidence_based_ood(detections)
            results.append((confidence_ood, conf_score, conf_reason))
            
            # Method 2: Class distribution analysis
            class_ood, class_score, class_reason = self._class_distribution_ood(detections)
            results.append((class_ood, class_score, class_reason))
            
            # Method 3: Image-level features analysis
            feature_ood, feature_score, feature_reason = self._feature_based_ood(frame)
            results.append((feature_ood, feature_score, feature_reason))
            
            # Combine results using weighted voting
            is_ood, final_score, final_reason = self._combine_ood_results(results)
            
            return is_ood, final_score, final_reason
            
        except Exception as e:
            self.logger.error(f"Error in OOD detection: {e}")
            return False, 0.0, f"Error in OOD detection: {str(e)}"
    
    def _confidence_based_ood(self, detections: List[Dict] = None) -> Tuple[bool, float, str]:
        """
        Detect OOD based on detection confidence scores.
        
        Args:
            detections: List of detection results
            
        Returns:
            tuple: (is_ood: bool, confidence: float, reason: str)
        """
        if not detections:
            return False, 1.0, "No detections to analyze"
        
        if len(detections) == 0:
            return False, 1.0, "No detections to analyze"
        
        # Calculate average confidence
        confidences = [det.get('confidence', 0.0) for det in detections]
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        # Check if average confidence is too low
        if avg_conf < self.confidence_threshold:
            return True, 1.0 - avg_conf, f"Low average confidence: {avg_conf:.3f}"
        
        return False, avg_conf, f"Acceptable average confidence: {avg_conf:.3f}"
    
    def _class_distribution_ood(self, detections: List[Dict] = None) -> Tuple[bool, float, str]:
        """
        Detect OOD based on class distribution analysis.
        
        Args:
            detections: List of detection results
            
        Returns:
            tuple: (is_ood: bool, confidence: float, reason: str)
        """
        if not detections:
            return False, 1.0, "No detections to analyze"
        
        if len(detections) == 0:
            return False, 1.0, "No detections to analyze"
        
        # Count known vs unknown classes
        known_count = 0
        unknown_count = 0
        
        for det in detections:
            fruit_class = det.get('fruit_class', '').lower()
            if fruit_class in [kc.lower() for kc in self.known_classes]:
                known_count += 1
            else:
                unknown_count += 1
        
        total_detections = len(detections)
        known_ratio = known_count / total_detections if total_detections > 0 else 0
        
        # Check if ratio of known classes is too low
        if known_ratio < self.min_detection_ratio:
            return True, 1.0 - known_ratio, f"Low known class ratio: {known_ratio:.3f}"
        
        return False, known_ratio, f"Acceptable known class ratio: {known_ratio:.3f}"
    
    def _feature_based_ood(self, frame: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect OOD based on image-level features.
        
        Args:
            frame: Input frame to analyze
            
        Returns:
            tuple: (is_ood: bool, confidence: float, reason: str)
        """
        try:
            # Analyze image complexity and texture
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate image variance (Laplacian for focus/blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate image entropy (measure of randomness)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            hist = hist[hist > 0]  # Remove zeros for log calculation
            entropy = -np.sum(hist * np.log2(hist))
            
            # Check if image is too blurry (low Laplacian variance)
            if laplacian_var < 50:  # Threshold for blur detection
                return True, 1.0, f"Image too blurry (Laplacian var: {laplacian_var:.2f})"
            
            # Check if image has too low entropy (too uniform)
            if entropy < 4.0:  # Threshold for texture analysis
                return True, 1.0, f"Image too uniform (Entropy: {entropy:.2f})"
            
            return False, 1.0, f"Acceptable image quality (Laplacian: {laplacian_var:.2f}, Entropy: {entropy:.2f})"
            
        except Exception as e:
            self.logger.error(f"Error in feature-based OOD: {e}")
            return False, 0.0, f"Error in feature analysis: {str(e)}"
    
    def _combine_ood_results(self, results: List[Tuple[bool, float, str]]) -> Tuple[bool, float, str]:
        """
        Combine multiple OOD detection results using weighted voting.
        
        Args:
            results: List of (is_ood, confidence, reason) tuples from different methods
            
        Returns:
            tuple: (is_ood: bool, confidence: float, reason: str)
        """
        if not results:
            return False, 1.0, "No OOD results to combine"
        
        # Count how many methods detected OOD
        ood_count = sum(1 for result in results if result[0])
        total_methods = len(results)
        
        # Calculate average confidence for OOD detection
        avg_confidence = np.mean([result[1] for result in results])
        
        # Determine if overall result is OOD (majority vote)
        is_ood = ood_count > total_methods / 2
        
        # Create combined reason
        reasons = [result[2] for result in results]
        combined_reason = f"OOD detection: {ood_count}/{total_methods} methods detected OOD. " + "; ".join(reasons)
        
        return is_ood, avg_confidence, combined_reason


def apply_ood_filtering(detections: List[Dict], ood_detector: OODDetector, frame: np.ndarray) -> List[Dict]:
    """
    Apply OOD filtering to detection results.
    
    Args:
        detections: List of detection results
        ood_detector: OOD detector instance
        frame: Input frame
        
    Returns:
        Filtered list of detections
    """
    if not ood_detector.enable_ood_detection:
        return detections
    
    # Check if the frame as a whole is OOD
    is_frame_ood, ood_conf, ood_reason = ood_detector.detect_ood(frame, detections)
    
    if is_frame_ood:
        logging.getLogger(__name__).info(f"Frame detected as OOD: {ood_reason}")
        # Return empty list if entire frame is OOD
        return []
    
    # Otherwise return original detections
    return detections