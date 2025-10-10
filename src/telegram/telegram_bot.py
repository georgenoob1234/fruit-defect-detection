"""
Telegram Bot module for the Fruit Defect Detection System.

This module handles sending notifications via Telegram bot.
"""

import logging
import requests
from pathlib import Path


class TelegramBot:
    """
    A class to handle Telegram bot operations for sending notifications.
    """
    
    def __init__(self, bot_token, users_config_path='config/telegram_users.yaml'):
        """
        Initialize the Telegram bot with the provided token.
        
        Args:
            bot_token (str): The Telegram bot token
            users_config_path (str): Path to the users config file
        """
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = logging.getLogger(__name__)
        self.users_config_path = users_config_path
        self.authorized_users = self._load_authorized_users()
        self.detection_logs = []  # Store detection logs
        self.last_update_id = 0  # For long polling

    def _load_authorized_users(self):
        """
        Load authorized users from the config file.
        
        Returns:
            set: Set of authorized user IDs
        """
        try:
            import yaml
            from pathlib import Path
            
            config_path = Path(self.users_config_path)
            if config_path.exists():
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    user_ids = config.get('telegram_users', {}).get('user_ids', [])
                    return set(user_ids)
            else:
                self.logger.warning(f"Users config file not found: {self.users_config_path}")
                return set()
        except Exception as e:
            self.logger.error(f"Error loading authorized users: {e}")
            return set()

    def _save_authorized_users(self):
        """
        Save authorized users to the config file.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            import yaml
            from pathlib import Path
            
            config_path = Path(self.users_config_path)
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare the config data
            config_data = {
                'telegram_users': {
                    'user_ids': sorted(list(self.authorized_users))
                }
            }
            
            # Write to file
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False)
            
            self.logger.info(f"Authorized users saved to {self.users_config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving authorized users: {e}")
            return False

    def is_authorized_user(self, user_id):
        """
        Check if a user is authorized to use admin commands.
        
        Args:
            user_id (int): The user ID to check
            
        Returns:
            bool: True if user is authorized, False otherwise
        """
        return user_id in self.authorized_users

    def add_user(self, user_id):
        """
        Add a user to the authorized users list.
        
        Args:
            user_id (int): The user ID to add
            
        Returns:
            bool: True if user was added successfully, False otherwise
        """
        try:
            self.authorized_users.add(user_id)
            return self._save_authorized_users()
        except Exception as e:
            self.logger.error(f"Error adding user {user_id}: {e}")
            return False

    def get_user_by_username(self, username):
        """
        Get user information by username using Telegram's getChat API.
        
        Args:
            username (str): The username to look up (with or without @)
            
        Returns:
            dict: User information if found, None otherwise
        """
        try:
            # Remove @ if present
            if username.startswith('@'):
                username = username[1:]
            
            response = requests.get(f"{self.base_url}/getChat", params={'chat_id': f'@{username}'})
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    return result.get('result')
                else:
                    self.logger.error(f"Failed to get user by username: {result.get('description')}")
                    return None
            else:
                self.logger.error(f"Failed to get user by username: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting user by username: {e}")
            return None

    def add_user_by_username(self, username):
        """
        Add a user by username.
        
        Args:
            username (str): The username to add (with or without @)
            
        Returns:
            bool: True if user was added successfully, False otherwise
        """
        try:
            user_info = self.get_user_by_username(username)
            if user_info:
                user_id = user_info.get('id')
                if user_id:
                    return self.add_user(user_id)
                else:
                    self.logger.error(f"No user ID found for username: {username}")
                    return False
            else:
                self.logger.error(f"User not found: {username}")
                return False
        except Exception as e:
            self.logger.error(f"Error adding user by username: {e}")
            return False

    def log_detection(self, detection_data):
        """
        Log a detection event.
        
        Args:
            detection_data (dict): Detection information to log
        """
        try:
            from datetime import datetime
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'detection_data': detection_data
            }
            self.detection_logs.append(log_entry)
            
            # Keep only the last 100 logs to prevent memory issues
            if len(self.detection_logs) > 100:
                self.detection_logs = self.detection_logs[-100:]
                
            self.logger.info(f"Detection logged: {detection_data}")
        except Exception as e:
            self.logger.error(f"Error logging detection: {e}")

    def get_detection_logs(self, count=10):
        """
        Get the most recent detection logs.
        
        Args:
            count (int): Number of logs to return (default 10)
            
        Returns:
            list: List of detection logs
        """
        try:
            return self.detection_logs[-count:] if len(self.detection_logs) >= count else self.detection_logs[:]
        except Exception as e:
            self.logger.error(f"Error getting detection logs: {e}")
            return []

    def send_message(self, chat_id, message, image_path=None):
        """
        Send a message to a Telegram chat, optionally with an image.
        
        Args:
            chat_id (int): The chat ID to send the message to
            message (str): The message text to send
            image_path (str, optional): Path to an image file to send with the message
            
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        try:
            if image_path and Path(image_path).exists():
                # Send photo with caption
                with open(image_path, 'rb') as photo_file:
                    files = {'photo': photo_file}
                    data = {
                        'chat_id': chat_id,
                        'caption': message
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/sendPhoto",
                        files=files,
                        data=data
                    )
            else:
                # Send text-only message
                data = {
                    'chat_id': chat_id,
                    'text': message
                }
                
                response = requests.post(
                    f"{self.base_url}/sendMessage",
                    data=data
                )
            
            if response.status_code == 200:
                self.logger.info(f"Message sent successfully to chat {chat_id}")
                return True
            else:
                self.logger.error(f"Failed to send message to chat {chat_id}: {response.text}")
                return False
                
        except FileNotFoundError:
            self.logger.error(f"Image file not found: {image_path}")
            # Send text-only message instead
            return self._send_text_message(chat_id, message)
        except Exception as e:
            self.logger.error(f"Error sending message to chat {chat_id}: {e}")
            return False

    def _send_text_message(self, chat_id, message):
        """
        Send a text-only message to a Telegram chat.
        
        Args:
            chat_id (int): The chat ID to send the message to
            message (str): The message text to send
            
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        try:
            data = {
                'chat_id': chat_id,
                'text': message
            }
            
            response = requests.post(
                f"{self.base_url}/sendMessage",
                data=data
            )
            
            if response.status_code == 200:
                self.logger.info(f"Text message sent successfully to chat {chat_id}")
                return True
            else:
                self.logger.error(f"Failed to send text message to chat {chat_id}: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending text message to chat {chat_id}: {e}")
            return False

    def get_me(self):
        """
        Get information about the bot.
        
        Returns:
            dict: Bot information if successful, None otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/getMe")
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get bot info: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting bot info: {e}")
            return None

    def get_updates(self):
        """
        Get updates for the bot (not used in this implementation).
        
        Returns:
            dict: Updates if successful, None otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/getUpdates")
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get updates: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting updates: {e}")
            return None

    def handle_command(self, user_id, command, args):
        """
        Handle incoming commands from users.
        
        Args:
            user_id (int): The ID of the user who sent the command
            command (str): The command text (e.g., 'start', 'adduser')
            args (list): List of arguments for the command
            
        Returns:
            str: Response message to send back to the user
        """
        # Check if user is authorized for admin commands
        is_authorized = self.is_authorized_user(user_id)
        
        if command == 'start':
            return self._handle_start_command(user_id)
        elif command == 'help':
            return self._handle_help_command(user_id)
        elif command == 'adduser':
            if is_authorized:
                return self._handle_adduser_command(user_id, args)
            else:
                return "❌ You are not authorized to use this command."
        elif command == 'showlogs':
            if is_authorized:
                return self._handle_showlogs_command(user_id, args)
            else:
                return "❌ You are not authorized to use this command."
        else:
            return "❌ Unknown command. Use /help to see available commands."

    def _handle_start_command(self, user_id):
        """
        Handle the /start command.
        
        Args:
            user_id (int): The ID of the user who sent the command
            
        Returns:
            str: Response message
        """
        welcome_msg = (
            "🍎 Welcome to the Fruit Defect Detection Bot! 🍎\n\n"
            "I'm here to help you monitor fruit quality in real-time.\n\n"
            "Available commands:\n"
            "/help - Show this help message\n"
            "/showlogs - Show recent detection logs (admin only)\n\n"
            f"Your user ID: {user_id}\n"
        )
        
        # Check if user is authorized
        if self.is_authorized_user(user_id):
            welcome_msg += "✅ You are an authorized user with admin privileges."
        else:
            welcome_msg += "ℹ️ You are not authorized for admin commands. Contact an admin to add you."
            
        return welcome_msg

    def _handle_help_command(self, user_id):
        """
        Handle the /help command.
        
        Args:
            user_id (int): The ID of the user who sent the command
            
        Returns:
            str: Response message
        """
        help_msg = (
            "🍎 Fruit Defect Detection Bot - Help 🍎\n\n"
            "Available commands:\n"
            "/start - Start interacting with the bot\n"
            "/help - Show this help message\n"
        )
        
        if self.is_authorized_user(user_id):
            help_msg += (
                "/adduser <username|user_id> - Add a user by username or user ID (admin only)\n"
                "/showlogs [count] - Show recent detection logs (admin only)\n"
            )
        else:
            help_msg += (
                "\nℹ️ You are not authorized for admin commands. Contact an admin to add you.\n"
            )
            
        help_msg += (
            "\nFor any issues, contact the system administrator."
        )
        
        return help_msg

    def _handle_adduser_command(self, user_id, args):
        """
        Handle the /adduser command.
        
        Args:
            user_id (int): The ID of the user who sent the command
            args (list): List of arguments for the command
            
        Returns:
            str: Response message
        """
        if not args:
            return "❌ Usage: /adduser <username|user_id>\nExample: /adduser @username or /adduser 123456789"
        
        target = args[0]
        
        # Check if it's a username (starts with @) or a numeric user ID
        if target.startswith('@') or target.lstrip('-').isdigit():
            if target.startswith('@'):
                # Add user by username
                success = self.add_user_by_username(target)
                if success:
                    return f"✅ Successfully added user {target} to authorized list."
                else:
                    return f"❌ Failed to add user {target}. Make sure the username exists and is correct."
            else:
                # Add user by ID
                try:
                    user_id_to_add = int(target)
                    success = self.add_user(user_id_to_add)
                    if success:
                        return f"✅ Successfully added user ID {user_id_to_add} to authorized list."
                    else:
                        return f"❌ Failed to add user ID {user_id_to_add}."
                except ValueError:
                    return f"❌ Invalid user ID: {target}. User ID must be a number."
        else:
            return f"❌ Invalid format: {target}. Use @username or numeric user ID."

    def _handle_showlogs_command(self, user_id, args):
        """
        Handle the /showlogs command.
        
        Args:
            user_id (int): The ID of the user who sent the command
            args (list): List of arguments for the command
            
        Returns:
            str: Response message
        """
        count = 10  # Default number of logs to show
        
        if args:
            try:
                count = int(args[0])
                if count <= 0:
                    count = 10
                elif count > 50:  # Limit max logs to prevent spam
                    count = 50
            except ValueError:
                return f"❌ Invalid number: {args[0]}. Please provide a valid number."
        
        logs = self.get_detection_logs(count)
        
        if not logs:
            return "📝 No detection logs available."
        
        response = f"📝 Recent Detection Logs (showing last {len(logs)}):\n\n"
        
        for i, log in enumerate(reversed(logs)):
            detection = log['detection_data']
            timestamp = log['timestamp']
            image_path = detection.get('image_path', 'N/A')
            response += (
                f"Detection #{len(logs)-i}:\n"
                f"  🍎 Fruit: {detection.get('fruit_class', 'Unknown')}\n"
                f"  ❌ Defective: {'Yes' if detection.get('is_defective', False) else 'No'}\n"
                f"  📊 Confidence: {detection.get('confidence', 0):.2f}\n"
                f"  📸 Image: {image_path}\n"
                f"  ⏰ Time: {timestamp}\n\n"
            )
        
        return response

    def process_updates(self):
        """
        Process incoming updates using long polling.
        
        Returns:
            list: List of processed updates
        """
        try:
            # Get updates with the last update ID to avoid processing old updates
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 20  # 20 second timeout for long polling
            }
            
            response = requests.get(f"{self.base_url}/getUpdates", params=params)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('ok'):
                    updates = result.get('result', [])
                    
                    for update in updates:
                        # Update the last update ID to prevent processing the same update again
                        self.last_update_id = max(self.last_update_id, update.get('update_id', 0))
                        
                        # Process the update
                        self._process_single_update(update)
                    
                    return updates
                else:
                    self.logger.error(f"Failed to get updates: {result.get('description')}")
                    return []
            else:
                self.logger.error(f"Failed to get updates: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error processing updates: {e}")
            return []

    def _process_single_update(self, update):
        """
        Process a single update from Telegram.
        
        Args:
            update (dict): The update object from Telegram API
        """
        try:
            # Check if the update contains a message
            if 'message' in update:
                message = update['message']
                chat_id = message['chat']['id']
                user_id = message['from']['id'] if 'from' in message else chat_id
                text = message.get('text', '')
                
                # Check if it's a command
                if text.startswith('/'):
                    # Parse the command and arguments
                    parts = text.split(' ', 1)
                    command = parts[0][1:].lower()  # Remove the '/' and convert to lowercase
                    args = parts[1].split() if len(parts) > 1 else []
                    
                    # Handle the command
                    response = self.handle_command(user_id, command, args)
                    
                    # Send response back to the user
                    self.send_message(chat_id, response)
                else:
                    # Not a command, send a helpful message
                    help_message = "ℹ️ I only respond to commands. Use /help to see available commands."
                    self.send_message(chat_id, help_message)
            else:
                # Update doesn't contain a message, might be other types of updates
                self.logger.debug(f"Received non-message update: {update}")
        except Exception as e:
            self.logger.error(f"Error processing update: {e}")

    def start_polling(self):
        """
        Start the long polling loop to continuously receive commands.
        This method runs indefinitely until interrupted.
        """
        self.logger.info("Starting Telegram bot polling...")
        
        import time
        try:
            while True:
                try:
                    # Process any new updates
                    self.process_updates()
                    
                    # Small delay to prevent excessive API calls
                    time.sleep(1)
                except KeyboardInterrupt:
                    self.logger.info("Polling interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in polling loop: {e}")
                    time.sleep(5)  # Wait 5 seconds before retrying
        except Exception as e:
            self.logger.error(f"Critical error in polling: {e}")
        finally:
            self.logger.info("Telegram bot polling stopped")

    def run_bot_in_thread(self):
        """
        Run the bot polling in a separate thread.
        
        Returns:
            threading.Thread: The thread object running the bot
        """
        import threading
        thread = threading.Thread(target=self.start_polling, daemon=True)
        thread.start()
        return thread