"""
Image utilities for fruit defect detection system with specialized augmentation for segmentation tasks.

This module provides functions for data augmentation that properly handle segmentation masks
and preserve mask alignment during transformations. It includes specialized augmentation for
handling glare, shadows, and blur as required by the technical specification.
"""
import cv2
import numpy as np
import random
from typing import Tuple, Optional, Union
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER
import albumentations as A


class SegmentationAugmentation:
    """
    Advanced augmentation class specifically designed for segmentation tasks that ensures
    proper alignment between images and their corresponding segmentation masks.
    """
    
    def __init__(self, 
                 img_size: int = 640,
                 hgain: float = 0.5,  # HSV gain for hue
                 sgain: float = 0.5,  # HSV gain for saturation
                 vgain: float = 0.5,  # HSV gain for value
                 degrees: float = 0.0,
                 translate: float = 0.1,
                 scale: float = 0.5,
                 shear: float = 0.0,
                 perspective: float = 0.0,
                 flipud: float = 0.0,
                 fliplr: float = 0.5,
                 mosaic: float = 1.0,
                 mixup: float = 0.0,
                 copy_paste: float = 0.0,
                 auto_augment: Optional[str] = None,
                 crop_fraction: float = 1.0):
        """
        Initialize segmentation-aware augmentation pipeline.
        
        Args:
            img_size: Target image size
            hgain, sgain, vgain: HSV augmentation gains
            degrees: Rotation degrees
            translate: Translation fraction
            scale: Scaling factor
            shear: Shear factor
            perspective: Perspective factor
            flipud: Vertical flip probability
            fliplr: Horizontal flip probability
            mosaic: Mosaic probability
            mixup: Mixup probability
            copy_paste: Copy-paste probability
            auto_augment: Auto augmentation policy (e.g., 'randaugment', 'autoaugment')
            crop_fraction: Fraction for random crops
        """
        self.img_size = img_size
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic = mosaic
        self.mixup = mixup
        self.copy_paste = copy_paste
        self.auto_augment = auto_augment
        self.crop_fraction = crop_fraction
        
        # Initialize Albumentations for advanced augmentation
        self.albumentations = None
        if auto_augment:
            self._init_albumentations()
    
    def _init_albumentations(self):
        """Initialize Albumentations for advanced augmentation."""
        # Define advanced augmentation transforms that work with both images and masks
        transforms = [
            A.HorizontalFlip(p=self.fliplr),
            A.VerticalFlip(p=self.flipud),
            A.ShiftScaleRotate(
                shift_limit=0.1 * self.translate,
                scale_limit=0.1 * self.scale,
                rotate_limit=int(self.degrees),
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2 * self.vgain,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(15 * self.hgain),
                sat_shift_limit=int(25 * self.sgain),
                val_shift_limit=int(25 * self.vgain),
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=(3, 7), p=0.1),
            A.MotionBlur(blur_limit=(3, 7), p=0.1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),
        ]
        
        self.albumentations = A.Compose(transforms)
    
    def apply_augmentation(self, img: np.ndarray, masks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentation to image and corresponding masks, ensuring alignment.
        
        Args:
            img: Input image (H, W, C)
            masks: Segmentation masks (H, W, N) where N is number of masks
            
        Returns:
            Tuple of (augmented_image, augmented_masks)
        """
        original_img = img.copy()
        original_masks = masks.copy() if masks is not None else None
        
        # Apply albumentations if available
        if self.albumentations is not None:
            if masks is not None:
                # Albumentations can handle both image and masks simultaneously
                transformed = self.albumentations(image=img, masks=masks)
                img = transformed['image']
                masks = transformed['masks']
            else:
                transformed = self.albumentations(image=img)
                img = transformed['image']
        
        return img, masks


def apply_glare_augmentation(img: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """
    Apply glare augmentation to simulate bright spots in images.
    
    Args:
        img: Input image
        intensity: Intensity of glare effect (0.0 to 1.0)
        
    Returns:
        Image with glare effect applied
    """
    img = img.astype(np.float32)
    
    # Create random bright spots
    h, w = img.shape[:2]
    num_spots = random.randint(1, 3)
    
    for _ in range(num_spots):
        # Random position
        x = random.randint(0, w)
        y = random.randint(0, h)
        
        # Random size and intensity
        radius = random.randint(5, int(min(h, w) * 0.1))
        strength = random.uniform(0.3, 1.0) * intensity
        
        # Create glare spot using Gaussian
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2
        glare = np.zeros_like(img)
        glare[mask] = [255, 255, 200]  # Bright yellowish glare
        
        # Apply glare with decreasing intensity away from center
        dist = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
        falloff = np.clip(1 - dist / radius, 0, 1)
        falloff = np.expand_dims(falloff, axis=-1)  # For broadcasting with channels
        glare = glare * falloff * strength
        
        img = np.clip(img + glare, 0, 255)
    
    return img.astype(np.uint8)


def apply_shadow_augmentation(img: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """
    Apply shadow augmentation to simulate shadowed regions in images.
    
    Args:
        img: Input image
        intensity: Intensity of shadow effect (0.0 to 1.0)
        
    Returns:
        Image with shadow effect applied
    """
    img = img.astype(np.float32)
    
    h, w = img.shape[:2]
    
    # Create random polygonal shadow areas
    num_shadows = random.randint(1, 2)
    
    for _ in range(num_shadows):
        # Define random polygon points for shadow
        num_points = random.randint(3, 6)
        points = []
        center_x, center_y = random.randint(0, w), random.randint(0, h)
        radius = random.randint(int(0.1 * min(h, w)), int(0.4 * min(h, w)))
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points + random.uniform(-0.5, 0.5)
            r = radius * random.uniform(0.7, 1.0)
            px = center_x + int(r * np.cos(angle))
            py = center_y + int(r * np.sin(angle))
            points.append([max(0, min(w-1, px)), max(0, min(h-1, py))])
        
        if len(points) >= 3:
            # Create shadow mask
            shadow_mask = np.zeros((h, w), dtype=np.float32)
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(shadow_mask, [points], 1.0)
            
            # Apply shadow effect (darken the image)
            shadow_strength = random.uniform(0.2, 0.6) * intensity
            shadow_effect = 1.0 - shadow_mask * shadow_strength
            shadow_effect = np.expand_dims(shadow_effect, axis=-1)  # For broadcasting with channels
            
            img = img * shadow_effect
    
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_blur_augmentation(img: np.ndarray, blur_type: str = 'motion', intensity: float = 0.3) -> np.ndarray:
    """
    Apply blur augmentation to simulate different types of blur.
    
    Args:
        img: Input image
        blur_type: Type of blur ('motion', 'gaussian', 'defocus', 'zoom')
        intensity: Intensity of blur effect (0.0 to 1.0)
        
    Returns:
        Image with blur effect applied
    """
    h, w = img.shape[:2]
    
    # Scale kernel size based on image dimensions and intensity
    kernel_size = max(1, int(min(h, w) * 0.02 * intensity * 10))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    if blur_type == 'motion':
        # Motion blur
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        img = cv2.filter2D(img, -1, kernel)
    elif blur_type == 'gaussian':
        # Gaussian blur
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif blur_type == 'defocus':
        # Defocus blur using disk kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = (kernel_size - 1) / 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i - center) ** 2 + (j - center) ** 2 <= (kernel_size / 2) ** 2:
                    kernel[i, j] = 1
        kernel = kernel / np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
    elif blur_type == 'zoom':
        # Zoom blur effect
        center = (w // 2, h // 2)
        img = img.astype(np.float32)
        for i in range(1, kernel_size + 1):
            # Create progressively smaller versions of the image
            scaled_img = cv2.resize(img, (w - i, h - i))
            padded_img = cv2.resize(scaled_img, (w, h))
            img = img * (1 - 1.0/kernel_size) + padded_img * (1.0/kernel_size)
        img = np.clip(img, 0, 255)
    
    return img.astype(np.uint8)


def apply_environmental_augmentation(img: np.ndarray, 
                                   apply_glare: bool = True,
                                   apply_shadows: bool = True, 
                                   apply_blur: bool = True,
                                   intensity: float = 0.3) -> np.ndarray:
    """
    Apply environmental augmentations (glare, shadows, blur) to simulate real-world conditions.
    
    Args:
        img: Input image
        apply_glare: Whether to apply glare augmentation
        apply_shadows: Whether to apply shadow augmentation
        apply_blur: Whether to apply blur augmentation
        intensity: Overall intensity of augmentations (0.0 to 1.0)
        
    Returns:
        Image with environmental augmentations applied
    """
    augmented_img = img.copy()
    
    # Apply augmentations in random order to create varied effects
    augmentations = []
    if apply_glare:
        augmentations.append(('glare', apply_glare_augmentation))
    if apply_shadows:
        augmentations.append(('shadows', apply_shadow_augmentation))
    if apply_blur:
        # Randomly select blur type
        blur_types = ['motion', 'gaussian', 'defocus', 'zoom']
        blur_type = random.choice(blur_types)
        augmentations.append(('blur', lambda img, i: apply_blur_augmentation(img, blur_type, i)))
    
    # Shuffle the augmentations
    random.shuffle(augmentations)
    
    for aug_name, aug_func in augmentations:
        augmented_img = aug_func(augmented_img, intensity * random.uniform(0.7, 1.3))
    
    return augmented_img


def resize_and_pad_image(img: np.ndarray, 
                        masks: Optional[np.ndarray] = None,
                        target_size: Union[int, Tuple[int, int]] = 640) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float], Tuple[int, int]]:
    """
    Resize and pad image maintaining aspect ratio, with corresponding mask transformation.
    
    Args:
        img: Input image (H, W, C)
        masks: Segmentation masks (H, W, N) or None
        target_size: Target size (int for square, or (width, height))
        
    Returns:
        Tuple of (resized_img, resized_masks, (scale_w, scale_h), (pad_w, pad_h))
    """
    if isinstance(target_size, int):
        target_w = target_h = target_size
    else:
        target_w, target_h = target_size
    
    h, w = img.shape[:2]
    
    # Calculate scale to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad image to target size
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    pad_w2 = target_w - new_w - pad_w
    pad_h2 = target_h - new_h - pad_h
    
    padded_img = cv2.copyMakeBorder(
        resized_img, pad_h, pad_h2, pad_w, pad_w2, 
        cv2.BORDER_CONSTANT, value=(114, 14, 114)  # YOLO default padding color
    )
    
    # Process masks if provided
    padded_masks = None
    if masks is not None:
        # Handle both single mask and multiple masks
        if len(masks.shape) == 2:  # Single mask
            resized_mask = cv2.resize(masks.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            padded_masks = cv2.copyMakeBorder(
                resized_mask, pad_h, pad_h2, pad_w, pad_w2, 
                cv2.BORDER_CONSTANT, value=0 # Black padding for masks
            )
            padded_masks = (padded_masks > 0.5).astype(np.uint8)  # Threshold back to binary
        else:  # Multiple masks (H, W, N)
            padded_masks = np.zeros((target_h, target_w, masks.shape[2]), dtype=masks.dtype)
            for i in range(masks.shape[2]):
                resized_mask = cv2.resize(
                    masks[:, :, i].astype(np.float32), 
                    (new_w, new_h), 
                    interpolation=cv2.INTER_NEAREST
                )
                padded_mask = cv2.copyMakeBorder(
                    resized_mask, pad_h, pad_h2, pad_w, pad_w2, 
                    cv2.BORDER_CONSTANT, value=0
                )
                padded_masks[:, :, i] = (padded_mask > 0.5).astype(padded_masks.dtype)
    
    return padded_img, padded_masks, (scale, scale), (pad_w, pad_h)


def normalize_image(img: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalize image pixel values using mean and standard deviation.
    
    Args:
        img: Input image (H, W, C) with values in range [0, 255]
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image with values in range [0, 1] or [-1, 1] depending on the normalization
    """
    img = img.astype(np.float32)
    img = img / 255.0  # Scale to [0, 1]
    
    # Apply normalization per channel
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
    
    return img


def denormalize_image(img: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Denormalize image from normalized values back to [0, 255] range.
    
    Args:
        img: Normalized image (H, W, C)
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized image with values in range [0, 255]
    """
    img = img.copy().astype(np.float32)
    
    # Apply denormalization per channel
    for i in range(img.shape[2]):
        img[:, :, i] = img[:, :, i] * std[i] + mean[i]
    
    img = img * 255.0  # Scale back to [0, 255]
    return np.clip(img, 0, 255).astype(np.uint8)


def convert_mask_to_polygon(mask: np.ndarray) -> list:
    """
    Convert binary segmentation mask to polygon coordinates.
    
    Args:
        mask: Binary segmentation mask (H, W) with values 0 or 1
        
    Returns:
        List of polygon points [(x1, y1), (x2, y2), ...]
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return empty list
    if not contours:
        return []
    
    # Get the largest contour (assuming it's the main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour to reduce number of points
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of tuples
    polygon_points = [(point[0][0], point[0][1]) for point in approx_contour]
    
    return polygon_points


def convert_polygon_to_mask(polygon: list, height: int, width: int) -> np.ndarray:
    """
    Convert polygon coordinates to binary segmentation mask.
    
    Args:
        polygon: List of polygon points [(x1, y1), (x2, y2), ...]
        height: Height of the output mask
        width: Width of the output mask
        
    Returns:
        Binary segmentation mask (H, W) with values 0 or 1
    """
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert polygon to the format expected by OpenCV
    polygon_array = np.array(polygon, dtype=np.int32)
    
    # Fill the polygon area with 1s
    cv2.fillPoly(mask, [polygon_array], 1)
    
    return mask


def mask_to_yolo_format(mask: np.ndarray) -> list:
    """
    Convert binary segmentation mask to YOLO segmentation format (normalized coordinates).
    
    Args:
        mask: Binary segmentation mask (H, W) with values 0 or 1
        
    Returns:
        List of normalized coordinates [x1, y1, x2, y2, ...] for the segmentation polygon
    """
    polygon_points = convert_mask_to_polygon(mask)
    
    if not polygon_points:
        return []
    
    # Get image dimensions
    h, w = mask.shape
    
    # Normalize coordinates to [0, 1] range
    normalized_coords = []
    for x, y in polygon_points:
        normalized_x = x / w
        normalized_y = y / h
        normalized_coords.extend([normalized_x, normalized_y])
    
    return normalized_coords


def yolo_to_mask_format(yolo_coords: list, img_height: int, img_width: int) -> np.ndarray:
    """
    Convert YOLO segmentation format (normalized coordinates) to binary mask.
    
    Args:
        yolo_coords: List of normalized coordinates [x1, y1, x2, y2, ...]
        img_height: Height of the original image
        img_width: Width of the original image
        
    Returns:
        Binary segmentation mask (H, W) with values 0 or 1
    """
    # Convert normalized coordinates back to pixel coordinates
    polygon_points = []
    for i in range(0, len(yolo_coords), 2):
        if i + 1 < len(yolo_coords):
            x = int(yolo_coords[i] * img_width)
            y = int(yolo_coords[i + 1] * img_height)
            polygon_points.append((x, y))
    
    # Create mask from polygon
    mask = convert_polygon_to_mask(polygon_points, img_height, img_width)
    
    return mask


def visualize_augmentation_results(original_img: np.ndarray, 
                                 augmented_img: np.ndarray, 
                                 original_mask: Optional[np.ndarray] = None,
                                 augmented_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a visualization showing original and augmented images/masks side by side.
    
    Args:
        original_img: Original image
        augmented_img: Augmented image
        original_mask: Original mask (optional)
        augmented_mask: Augmented mask (optional)
        
    Returns:
        Combined visualization image
    """
    # Ensure images are the same size
    h, w = original_img.shape[:2]
    aug_h, aug_w = augmented_img.shape[:2]
    
    if h != aug_h or w != aug_w:
        augmented_img = cv2.resize(augmented_img, (w, h))
        if augmented_mask is not None:
            augmented_mask = cv2.resize(augmented_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create mask overlays if masks are provided
    def create_mask_overlay(img, mask):
        if mask is None:
            return img
        # Create a colored overlay for the mask
        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green for mask areas
        # Blend the overlay with the original image
        combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        return combined
    
    original_with_mask = create_mask_overlay(original_img, original_mask)
    augmented_with_mask = create_mask_overlay(augmented_img, augmented_mask)
    
    # Combine images horizontally
    combined = np.hstack([original_with_mask, augmented_with_mask])
    
    return combined