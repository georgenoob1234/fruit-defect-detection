"""
Post-processing utilities for segmentation masks in fruit defect detection system.

This module provides functions for refining segmentation masks, including
morphological operations, noise removal, and mask optimization.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging


class SegmentationPostProcessor:
    """
    A class to post-process segmentation masks for improved quality and accuracy.
    """
    
    def __init__(self, 
                 min_area_ratio: float = 0.001,  # Minimum area as ratio of image size
                 max_area_ratio: float = 0.9,    # Maximum area as ratio of image size
                 morph_kernel_size: int = 3,     # Size of morphological operation kernels
                 connectivity: int = 8,          # Connectivity for morphological operations
                 apply_closing: bool = True,     # Whether to apply morphological closing
                 apply_opening: bool = True,     # Whether to apply morphological opening
                 apply_smoothing: bool = True,   # Whether to apply smoothing
                 smoothing_kernel_size: int = 5, # Size of smoothing kernel
                 smoothing_sigma: float = 1.0    # Sigma for Gaussian smoothing
                 ):
        """
        Initialize the post-processor with configuration parameters.
        
        Args:
            min_area_ratio: Minimum area threshold as ratio of image size
            max_area_ratio: Maximum area threshold as ratio of image size
            morph_kernel_size: Size of kernel for morphological operations
            connectivity: Connectivity for morphological operations
            apply_closing: Whether to apply morphological closing
            apply_opening: Whether to apply morphological opening
            apply_smoothing: Whether to apply smoothing
            smoothing_kernel_size: Size of smoothing kernel
            smoothing_sigma: Sigma for Gaussian smoothing
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.morph_kernel_size = morph_kernel_size
        self.connectivity = connectivity
        self.apply_closing = apply_closing
        self.apply_opening = apply_opening
        self.apply_smoothing = apply_smoothing
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_sigma = smoothing_sigma
        
        # Create morphological operation kernels
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Segmentation post-processor initialized")
    
    def post_process_mask(self, 
                         mask: np.ndarray, 
                         image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Post-process a single segmentation mask.
        
        Args:
            mask: Binary segmentation mask (H, W) with values 0 or 1
            image_shape: Original image shape (H, W) for area calculations
            
        Returns:
            Post-processed binary mask
        """
        # Ensure mask is binary
        processed_mask = (mask > 0.5).astype(np.uint8)
        
        # Apply morphological operations
        if self.apply_closing:
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, self.kernel)
        
        if self.apply_opening:
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Apply smoothing if requested
        if self.apply_smoothing:
            # Convert to float for smoothing
            float_mask = processed_mask.astype(np.float32)
            # Apply Gaussian blur
            smoothed = cv2.GaussianBlur(
                float_mask, 
                (self.smoothing_kernel_size, self.smoothing_kernel_size), 
                self.smoothing_sigma
            )
            # Threshold back to binary
            processed_mask = (smoothed > 0.5).astype(np.uint8)
        
        # Filter connected components by area if image shape is provided
        if image_shape is not None:
            processed_mask = self._filter_by_area(processed_mask, image_shape)
        
        return processed_mask
    
    def post_process_masks(self, 
                          masks: List[np.ndarray], 
                          image_shape: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Post-process a list of segmentation masks.
        
        Args:
            masks: List of binary segmentation masks
            image_shape: Original image shape (H, W) for area calculations
            
        Returns:
            List of post-processed binary masks
        """
        processed_masks = []
        for mask in masks:
            processed_mask = self.post_process_mask(mask, image_shape)
            # Only add mask if it has content after processing
            if processed_mask.sum() > 0:
                processed_masks.append(processed_mask)
        
        return processed_masks
    
    def _filter_by_area(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Filter connected components by area.
        
        Args:
            mask: Binary mask to filter
            image_shape: Original image shape (H, W)
            
        Returns:
            Filtered binary mask
        """
        h, w = image_shape
        total_pixels = h * w
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=self.connectivity)
        
        # Calculate area thresholds
        min_area = int(self.min_area_ratio * total_pixels)
        max_area = int(self.max_area_ratio * total_pixels)
        
        # Create output mask
        filtered_mask = np.zeros_like(mask)
        
        # Process each connected component (skip background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Check if area is within acceptable range
            if min_area <= area <= max_area:
                # Add this component to the output mask
                filtered_mask[labels == i] = 1
        
        return filtered_mask
    
    def refine_mask_boundaries(self, 
                              mask: np.ndarray, 
                              original_image: np.ndarray,
                              edge_threshold: float = 0.1) -> np.ndarray:
        """
        Refine mask boundaries based on image edges.
        
        Args:
            mask: Binary segmentation mask
            original_image: Original image for edge computation
            edge_threshold: Threshold for edge detection
            
        Returns:
            Mask with refined boundaries
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Compute edges in the original image
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        
        # Create a distance transform to find mask boundaries
        dist_transform = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 5)
        
        # Find pixels near the mask boundary
        boundary_mask = (dist_transform < 3) & (dist_transform > 0)
        
        # Adjust mask based on edges
        refined_mask = binary_mask.copy()
        
        # For pixels near the boundary, check if there's a strong edge nearby
        edge_nearby = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > edge_threshold
        
        # Adjust the mask where edges are detected near boundaries
        refine_area = boundary_mask & edge_nearby
        refined_mask[refine_area] = 1 # Extend mask to edge
        
        # Apply morphological operations to clean up
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, self.kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, self.kernel)
        
        return refined_mask


def convert_raw_model_output_to_masks(raw_output,
                                     confidence_threshold: float = 0.5,
                                     mask_format: str = 'binary') -> List[np.ndarray]:
    """
    Convert raw model output to proper segmentation masks.
    
    Args:
        raw_output: Raw output from segmentation model
        confidence_threshold: Confidence threshold for mask binarization
        mask_format: Format of output masks ('binary' or 'probability')
        
    Returns:
        List of segmentation masks
    """
    masks = []
    
    # Handle different types of model outputs
    if hasattr(raw_output, 'masks') and raw_output.masks is not None:
        # YOLO model output with masks attribute
        if hasattr(raw_output.masks, 'data'):
            for mask_tensor in raw_output.masks.data:
                mask_np = mask_tensor.cpu().numpy()
                
                if mask_format == 'binary':
                    mask_np = (mask_np > confidence_threshold).astype(np.uint8)
                
                masks.append(mask_np)
        else:
            # Direct mask array
            for mask in raw_output.masks:
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                
                if mask_format == 'binary':
                    mask_np = (mask_np > confidence_threshold).astype(np.uint8)
                
                masks.append(mask_np)
    elif isinstance(raw_output, (list, tuple)):
        # List/tuple of masks
        for mask in raw_output:
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
            
            if mask_format == 'binary':
                mask_np = (mask_np > confidence_threshold).astype(np.uint8)
            
            masks.append(mask_np)
    elif isinstance(raw_output, np.ndarray):
        # Single array that might contain multiple masks
        if len(raw_output.shape) == 3:
            # Multiple masks stacked along first dimension
            for i in range(raw_output.shape[0]):
                mask_np = raw_output[i]
                
                if mask_format == 'binary':
                    mask_np = (mask_np > confidence_threshold).astype(np.uint8)
                
                masks.append(mask_np)
        else:
            # Single mask
            mask_np = raw_output
            
            if mask_format == 'binary':
                mask_np = (mask_np > confidence_threshold).astype(np.uint8)
            
            masks.append(mask_np)
    else:
        # Handle other types of raw outputs
        logging.getLogger(__name__).warning(f"Unknown raw output type: {type(raw_output)}, attempting conversion")
        try:
            mask_np = np.array(raw_output)
            if mask_format == 'binary':
                mask_np = (mask_np > confidence_threshold).astype(np.uint8)
            masks.append(mask_np)
        except Exception as e:
            logging.getLogger(__name__).error(f"Could not convert raw output to mask: {e}")
    
    return masks


class AdvancedSegmentationPostProcessor(SegmentationPostProcessor):
    """
    An advanced version of the segmentation post-processor with additional features.
    """
    
    def __init__(self,
                 min_area_ratio: float = 0.001,
                 max_area_ratio: float = 0.9,
                 morph_kernel_size: int = 3,
                 connectivity: int = 8,
                 apply_closing: bool = True,
                 apply_opening: bool = True,
                 apply_smoothing: bool = True,
                 smoothing_kernel_size: int = 5,
                 smoothing_sigma: float = 1.0,
                 apply_watershed: bool = False,  # Whether to apply watershed segmentation
                 watershed_threshold: float = 0.5  # Threshold for watershed pre-processing
                 ):
        """
        Initialize the advanced post-processor with configuration parameters.
        
        Args:
            min_area_ratio: Minimum area threshold as ratio of image size
            max_area_ratio: Maximum area threshold as ratio of image size
            morph_kernel_size: Size of kernel for morphological operations
            connectivity: Connectivity for morphological operations
            apply_closing: Whether to apply morphological closing
            apply_opening: Whether to apply morphological opening
            apply_smoothing: Whether to apply smoothing
            smoothing_kernel_size: Size of smoothing kernel
            smoothing_sigma: Sigma for Gaussian smoothing
            apply_watershed: Whether to apply watershed segmentation for separating touching objects
            watershed_threshold: Threshold for watershed pre-processing
        """
        super().__init__(
            min_area_ratio, max_area_ratio, morph_kernel_size, connectivity,
            apply_closing, apply_opening, apply_smoothing,
            smoothing_kernel_size, smoothing_sigma
        )
        
        self.apply_watershed = apply_watershed
        self.watershed_threshold = watershed_threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Advanced segmentation post-processor initialized")
    
    def post_process_mask(self,
                         mask: np.ndarray,
                         image_shape: Optional[Tuple[int, int]] = None,
                         original_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Post-process a single segmentation mask with advanced options.
        
        Args:
            mask: Binary segmentation mask (H, W) with values 0 or 1
            image_shape: Original image shape (H, W) for area calculations
            original_image: Original image for advanced processing techniques
            
        Returns:
            Post-processed binary mask
        """
        # Start with basic post-processing
        processed_mask = super().post_process_mask(mask, image_shape)
        
        # Apply watershed segmentation if enabled and original image is provided
        if self.apply_watershed and original_image is not None:
            processed_mask = self._apply_watershed(processed_mask, original_image)
        
        return processed_mask
    
    def _apply_watershed(self, mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Apply watershed segmentation to separate touching objects.
        
        Args:
            mask: Binary mask to process
            original_image: Original image for watershed processing
            
        Returns:
            Mask with separated objects
        """
        # Create an estimated background from the original image
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply distance transform to the mask
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Create a thresholded distance transform to identify object centers
        _, sure_fg = cv2.threshold(dist_transform,
                                  self.watershed_threshold * dist_transform.max(),
                                  255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Create a background mask
        unknown = cv2.subtract(mask, sure_fg)
        
        # Label the markers
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
        
        # Convert watershed result back to binary mask
        # Markers that are > 1 are the segmented objects
        watershed_mask = np.zeros_like(mask)
        watershed_mask[markers > 1] = 1
        
        return watershed_mask