# Detection Logging System Documentation

This document explains the detection logging system in the Fruit Defect Detection System.

## Overview

The detection logging system maintains a record of all detection events in the application. This system is integrated with the Telegram bot functionality and allows for retrieval of recent detection logs via the `/showlogs` command.

## Features

### 1. Automatic Detection Logging
- Every detection event is automatically logged when processed by the system (no deduplication applied to logging)
- Each log entry includes:
  - Timestamp of the detection
  - Detection data (fruit class, defective status, confidence, bounding box, image path)
- The system maintains the last 100 logs to prevent memory issues
- Logging occurs regardless of whether notifications are sent (based on deduplication logic)

### 2. Log Storage
- Detection logs are stored in memory within the `TelegramBot` class
- Only the most recent 100 logs are kept to prevent memory overflow
- Logs include complete detection information including image paths when available

### 3. Log Retrieval via Telegram
- Authorized users can retrieve logs using the `/showlogs [count]` command
- The optional count parameter specifies how many logs to retrieve (default is 10)
- Logs are displayed in reverse chronological order (most recent first)
- Each log entry shows fruit class, defective status, confidence, and timestamp

## Implementation Details

### `src/telegram/telegram_bot.py`
- `log_detection()` method: Stores detection events with timestamp
- `get_detection_logs()` method: Retrieves specified number of recent logs
- `_handle_showlogs_command()` method: Handles the /showlogs command
- The system stores detection logs in the `detection_logs` list attribute

## Data Structure

Each detection log entry contains:
```python
{
  'timestamp': '2023-12-01T10:30:45.123456',
  'detection_data': {
    'fruit_class': 'apple',
    'is_defective': True,
    'confidence': 0.85,
    'timestamp': '2023-12-01T10:30:45.123456',
    'bbox': [x1, y1, x2, y2],
    'image_path': '/path/to/captured/image.jpg'  # if available
  }
}
```

## Integration with Photo Capture

- All detected fruits (both defective and non-defective) are captured for logging purposes
- Image paths are included in the detection logs when available
- This provides complete documentation of all detection events

## Usage

### Retrieving Logs via Telegram
1. Ensure you are an authorized user (your ID is in `config/telegram_users.yaml`)
2. Send the `/showlogs` command to the bot
3. Optionally specify a count: `/showlogs 5` to retrieve 5 most recent logs

## Security Considerations

- Only authorized users can access the logs via the Telegram bot commands
- The system does not expose logs through other interfaces without authentication
- Log data is stored in memory and will be cleared when the application restarts

## Troubleshooting

### No Logs Available
- Ensure the system is actively detecting fruits
- Check that detection events are being processed successfully
- Verify the application has been running long enough to generate logs

### Logs Don't Show Images
- Verify that photo capture is working correctly
- Check that the `captured_images/` directory exists and is writable
- Ensure the image paths in the logs are accessible