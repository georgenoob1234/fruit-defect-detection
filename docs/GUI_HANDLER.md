# GUI Handler Documentation

## Overview

The GUI Handler module provides a graphical user interface for the Fruit Defect Detection System. It displays the camera feed with bounding boxes around detected objects, class names, and defect status.

## Components

### GUIHandler Class

The `GUIHandler` class manages the graphical user interface for the fruit defect detection system. It displays camera feed with bounding boxes, class names, and defect status.

#### Methods

- `__init__(window_name="Fruit Defect Detection")`: Initializes the GUI handler with a specified window name.
- `draw_detections(frame, detection_results)`: Draws detection results on the frame with bounding boxes, labels, and confidence bars.
- `show_frame(frame, detection_results=None)`: Displays the frame with detection results and returns a boolean indicating whether to continue running.
- `cleanup()`: Cleans up GUI resources.
- `set_window_size(width, height)`: Sets the window size.

#### Features

1. **Bounding Boxes**: Draws bounding boxes around detected fruits with color coding:
   - Green for normal fruits
   - Red for defective fruits

2. **Defect Segmentation**: When a fruit is detected as defective, overlays segmentation masks to show the exact location of defects.

3. **Labels**: Shows fruit class, defect status, and confidence score on each detection.

4. **Confidence Bars**: Displays a visual confidence bar below each bounding box.

5. **Information Panel**: Shows overall detection statistics including:
   - Total number of detections
   - Number of defective fruits
   - Application title and exit instructions

6. **Color Coding**: Uses intuitive color coding to quickly identify defective vs normal fruits.

## Integration

The GUI handler is integrated with the main application loop and works with the existing detection pipeline. Both `main.py` and `src/main_loop.py` have been updated to use the GUI handler instead of manual drawing functions.

## Configuration

GUI functionality is controlled through the `app_config.yaml` file:

```yaml
# GUI Settings
gui:
  show_gui: true  # Set to false to disable GUI
  show_defect_status: true  # When true, shows both fruit type and defect status; when false, shows only fruit type
```

## Usage

When the application runs with GUI enabled, it will display a window showing:
- The live camera feed
- Bounding boxes around detected fruits
- Labels indicating fruit type and defect status
- Confidence scores for each detection
- Overall detection statistics

Press 'Q' to quit the application.

## Dependencies

- OpenCV (cv2)
- NumPy
- Python standard library modules: logging, typing

## Example Detection Result Format

The GUI handler expects detection results in the following format:

```python
detection_result = {
    'fruit_class': 'apple',      # Type of fruit detected
    'is_defective': True,        # Whether the fruit is defective
    'confidence': 0.95,          # Confidence score (0.0-1.0)
    'bbox': [x1, y1, x2, y2]     # Bounding box coordinates
}
```

Multiple detection results are passed as a list to the `draw_detections` method.