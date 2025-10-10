# Telegram Bot Integration Documentation

This document explains the Telegram bot integration in the Fruit Defect Detection System.

## Overview

The application integrates with Telegram to send real-time notifications when fruits are detected. Notifications include detection details and captured images of defective fruits. The system also supports enhanced bot commands for administration and monitoring.

## Enhanced Commands

The Telegram bot now supports several commands for interaction and monitoring:

### User Commands
- `/start` - Start interacting with the bot and get welcome message
- `/help` - Show available commands and usage information

### Admin Commands (Authorized Users Only)
- `/adduser <username|user_id>` - Add a new user by username or user ID (admin only)
- `/showlogs [count]` - Show recent detection logs (admin only, default count is 10)

## Setup

### 1. Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Start a chat with BotFather and use the `/newbot` command
3. Follow the instructions to name your bot and get the API token
4. Copy the bot token provided by BotFather

### 2. Configure the Bot Token

Update `config/telegram_config.yaml` with your bot token:

```yaml
# Telegram Bot Configuration for Fruit Defect Detection System

# Telegram Bot Settings
telegram:
  bot_token: "YOUR_TELEGRAM_BOT_TOKEN_HERE"  # Replace with your actual bot token
  enable_telegram: true  # Set to false to disable Telegram bot functionality
```

### 3. Get User IDs

To get the Telegram user IDs for authorized users:

1. Search for `@userinfobot` in Telegram
2. Start a chat with the bot
3. It will provide your user ID
4. Alternatively, you can use webhooks or bot commands to capture user IDs

### 4. Configure Authorized Users

Update `config/telegram_users.yaml` with the authorized user IDs:

```yaml
# Authorized Telegram User IDs for Fruit Defect Detection System

# List of authorized user IDs who can receive notifications
telegram_users:
 user_ids: 
    - 123456789  # Replace with actual Telegram user ID
    - 987654321  # Replace with actual Telegram user ID
```

## Data Sent in Notifications

When a detection occurs, the system sends the following information to Telegram:

```
🍎 Fruit Detection Alert 🍎
Fruit: apple
Defective: Yes
Confidence: 0.85
Time: 2023-12-01T10:30:45.123456
```

If the detected fruit is defective, an image of the fruit is also sent with the message.

## How the Sending Mechanism Works

### 1. Detection Processing
- When a fruit is detected and processed by the system
- The debouncing mechanism checks if enough time has passed since the last notification for the same fruit class
- If the fruit is defective and the image is captured, the image path is stored

### 2. Message Preparation
- The system prepares a formatted message with detection details
- If an image was captured (for defective fruits), it's included with the message

### 3. Notification Sending
- The system iterates through all authorized user IDs in the configuration
- For each user ID, it sends the message (with image if available) using the Telegram Bot API
- Success or failure is logged for each notification attempt

### 4. Error Handling
- If the image file is not found, the system sends the text message only
- Network errors are caught and logged
- Each user receives the notification independently, so one failure doesn't affect others

## Configuration Options

The Telegram integration is configured in two files:

### `config/telegram_config.yaml`
- `bot_token`: Your Telegram bot token from BotFather
- `enable_telegram`: Boolean flag to enable or disable Telegram bot functionality (default: true)

### `config/telegram_users.yaml`
- `user_ids`: List of authorized user IDs who can receive notifications

## Implementation Details

The Telegram functionality is implemented in `src/telegram/telegram_bot.py`:

- `TelegramBot` class handles all Telegram API interactions
- `send_message()` method sends text messages and images to specified chat IDs
- Uses the official Telegram Bot API via HTTPS requests
- Includes proper error handling and logging

## Security Considerations

- Only users with IDs listed in `telegram_users.yaml` can receive notifications
- The bot token should be kept secure and not shared publicly
- The bot only sends messages, it doesn't receive or process incoming messages

## Troubleshooting

### Bot Not Sending Messages
- Verify the bot token is correct in `config/telegram_config.yaml`
- Ensure user IDs in `config/telegram_users.yaml` are correct
- Check that users have started a chat with the bot (send any message to it first)
- Make sure the `enable_telegram` option is set to `true`

### Images Not Sending
- Verify that the `captured_images/` directory exists and is writable
- Check that image paths in the logs are correct
- Ensure the image files exist at the specified paths

### Configuration Issues
- Ensure YAML syntax is correct in configuration files
- Check that the application has read permissions for config files