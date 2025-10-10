# Web API Integration Documentation

This document provides specific instructions on the web API integration in the Fruit Defect Detection System.

## Overview

The application can send detection data to an external web API endpoint. This integration allows for remote monitoring, data collection, and integration with other systems. The API integration supports both detection data and image uploads.

## Configuration

API settings are configured in `config/app_config.yaml` under the `api` section:

```yaml
api:
  enabled: false  # Whether to enable API notifications (true/false)
  base_url: "http://localhost:8000"  # Base URL of the API
  endpoint: "/api/detections"  # API endpoint for sending detection data
 timeout: 10  # Request timeout in seconds
```

To enable API integration, set `enabled: true` and update the `base_url` and `endpoint` to match your API server.

## Data Format

When a detection occurs and API integration is enabled, the system sends the following data to the configured endpoint:

```json
{
  "fruit_class": "apple",
  "is_defective": true,
  "confidence": 0.85,
  "timestamp": "2023-12-01T10:30:45.123456",
  "bbox": [x1, y1, x2, y2]
}
```

### Fields Explanation

- `fruit_class`: The type of fruit detected (e.g., "apple", "banana", "tomato")
- `is_defective`: Boolean indicating if the fruit is defective
- `confidence`: Confidence score of the detection (0.0-1.0)
- `timestamp`: ISO format timestamp of the detection
- `bbox`: Array of bounding box coordinates [x1, y1, x2, y2]

## Image Upload

When a defective fruit is detected and captured, the image is sent along with the detection data. The system supports two methods of image upload:

### Method 1: Multipart Form Data (Recommended)
- Detection data is sent as JSON in the `detection_data` field
- Image is sent as a file in the `image` field
- Content-Type: `multipart/form-data`

### Method 2: JSON Only
- Only detection data is sent as JSON
- No image is uploaded
- Content-Type: `application/json`

## When Requests Are Sent

API requests are sent when:
1. A fruit is detected with sufficient confidence
2. The debouncing time has passed since the last detection of the same fruit class
3. API integration is enabled in the configuration
4. The detection processing is successful

## Implementation Details

The API requests are handled by the `APIHandler` class in `src/api/api_handler.py`. The handler:

1. Checks if an image path is provided
2. If an image is available, sends the request as multipart form data
3. If no image is available, sends the request as JSON
4. Includes proper error handling and timeouts
5. Logs the success or failure of each request

## Error Handling

The API handler includes robust error handling:
- Request timeouts are handled with a configurable timeout value
- Network errors are caught and logged
- HTTP error responses are logged for debugging
- If an image upload fails, the system falls back to sending text-only data

## Testing the Integration

To test the API integration:
1. Set up an API server that can receive POST requests at your configured endpoint
2. Enable API integration in `config/app_config.yaml`
3. Update the base URL and endpoint to match your server
4. Run the application and observe the logs for API request results