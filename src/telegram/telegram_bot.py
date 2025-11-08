"""
Telegram Bot module for the Fruit Defect Detection System.

This module handles sending notifications via Telegram bot using python-telegram-bot library.
"""
import logging
from pathlib import Path
import threading
import time
import queue
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Callable
from telegram import Update, Bot
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TelegramError
import yaml


@dataclass
class QueuedMessage:
    """
    Represents a message waiting to be sent with its metadata.
    """
    chat_id: int
    message: str
    image_path: Optional[str]
    timeout: int
    timestamp: float
    callback: Optional[Callable]


class TelegramBot:
    """
    A class to handle Telegram bot operations for sending notifications using python-telegram-bot.
    """
    
    def __init__(self, bot_token, users_config_path='config/telegram_users.yaml'):
        """
        Initialize the Telegram bot with the provided token.
        
        Args:
            bot_token (str): The Telegram bot token
            users_config_path (str): Path to the users config file
        """
        self.bot_token = bot_token
        self.logger = logging.getLogger(__name__)
        self.users_config_path = users_config_path
        self.authorized_users = self._load_authorized_users()
        self.detection_logs = []  # Store detection logs
        
        # Initialize the application
        self.application = ApplicationBuilder().token(bot_token).build()
        self.bot = self.application.bot
        
        # Load rate limiting configuration
        self._load_rate_limit_config()
        
        # Track last message time for rate limiting
        self.last_message_time = 0
        
        # Removed comprehensive rate limiting system
        # Now send messages immediately without any cooldown or rate limiting
        self.message_queue = queue.Queue()  # Queue for messages (though we'll send immediately now)
        self.queue_processor_thread = None # Thread to process the message queue
        self.queue_processor_running = False # Flag to control the queue processor
        
        # Start the queue processor thread
        self._start_queue_processor()
        
        # Add command handlers
        self._add_handlers()
        
    def _load_rate_limit_config(self):
        """
        Load rate limit configuration from the telegram config file.
        """
        try:
            # Load telegram config
            config_path = Path('config/telegram_config.yaml')
            if config_path.exists():
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    telegram_config = config.get('telegram', {})
                    self.rate_limit_seconds = telegram_config.get('rate_limit_seconds', 5)
                    self.logger.info(f"Rate limit loaded: {self.rate_limit_seconds} seconds")
            else:
                # Default rate limit if config file doesn't exist
                self.rate_limit_seconds = 5
                self.logger.warning(f"Config file {config_path} not found, using default rate limit of {self.rate_limit_seconds} seconds")
        except Exception as e:
            self.logger.error(f"Error loading rate limit config: {e}")
            self.rate_limit_seconds = 5  # Default to 5 seconds
            
    def _check_rate_limit(self):
        """
        Check if enough time has passed since the last message to respect the rate limit.
        
        Returns:
            bool: True if message can be sent, False if rate limit is exceeded
        """
        current_time = time.time()
        time_since_last_message = current_time - self.last_message_time
        
        if time_since_last_message < self.rate_limit_seconds:
            remaining_time = self.rate_limit_seconds - time_since_last_message
            self.logger.warning(f"Rate limit exceeded. Waiting {remaining_time:.2f}s before sending next message. Rate limit: {self.rate_limit_seconds}s")
            return False
            
        return True
        
    def _update_last_message_time(self):
        """
        Update the timestamp of the last message sent.
        """
        self.last_message_time = time.time()
        
    def _add_handlers(self):
        """Add command and message handlers to the application."""
        # Command handlers
        self.application.add_handler(CommandHandler('start', self._start_command))
        self.application.add_handler(CommandHandler('help', self._help_command))
        self.application.add_handler(CommandHandler('adduser', self._adduser_command))
        self.application.add_handler(CommandHandler('showlogs', self._showlogs_command))
        
        # Handle unknown commands
        self.application.add_handler(MessageHandler(filters.COMMAND, self._unknown_command))
        
        # Add error handler
        self.application.add_error_handler(self._error_handler)
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        user_id = update.effective_user.id
        
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: start")
            await update.message.reply_text("⏳ Please wait before sending another command. Rate limit exceeded.")
            return
            
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
        
        await update.message.reply_text(welcome_msg)
        # Update the last message time after sending the response
        self._update_last_message_time()
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command."""
        user_id = update.effective_user.id
        
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: help")
            await update.message.reply_text("⏳ Please wait before sending another command. Rate limit exceeded.")
            return
            
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
        
        await update.message.reply_text(help_msg)
        # Update the last message time after sending the response
        self._update_last_message_time()
    
    async def _adduser_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /adduser command."""
        user_id = update.effective_user.id
        
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: adduser")
            await update.message.reply_text("⏳ Please wait before sending another command. Rate limit exceeded.")
            return
        
        # Check if user is authorized
        if not self.is_authorized_user(user_id):
            await update.message.reply_text("❌ You are not authorized to use this command.")
            # Update the last message time after sending the response
            self._update_last_message_time()
            return
        
        if not context.args:
            await update.message.reply_text("❌ Usage: /adduser <username|user_id>\nExample: /adduser @username or /adduser 123456789")
            # Update the last message time after sending the response
            self._update_last_message_time()
            return
        
        target = context.args[0]
        
        # Check if it's a username (starts with @) or a numeric user ID
        if target.startswith('@') or target.lstrip('-').isdigit():
            if target.startswith('@'):
                # Add user by username
                success = self.add_user_by_username(target, timeout=30)
                if success:
                    await update.message.reply_text(f"✅ Successfully added user {target} to authorized list.")
                else:
                    await update.message.reply_text(f"❌ Failed to add user {target}. Make sure the username exists and is correct.")
            else:
                # Add user by ID
                try:
                    user_id_to_add = int(target)
                    success = self.add_user(user_id_to_add)
                    if success:
                        await update.message.reply_text(f"✅ Successfully added user ID {user_id_to_add} to authorized list.")
                    else:
                        await update.message.reply_text(f"❌ Failed to add user ID {user_id_to_add}.")
                except ValueError:
                    await update.message.reply_text(f"❌ Invalid user ID: {target}. User ID must be a number.")
        else:
            await update.message.reply_text(f"❌ Invalid format: {target}. Use @username or numeric user ID.")
            
        # Update the last message time after sending the response
        self._update_last_message_time()
    
    async def _showlogs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /showlogs command."""
        user_id = update.effective_user.id
        
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: showlogs")
            await update.message.reply_text("⏳ Please wait before sending another command. Rate limit exceeded.")
            return
        
        # Check if user is authorized
        if not self.is_authorized_user(user_id):
            await update.message.reply_text("❌ You are not authorized to use this command.")
            # Update the last message time after sending the response
            self._update_last_message_time()
            return
        
        count = 10  # Default number of logs to show
        
        if context.args:
            try:
                count = int(context.args[0])
                if count <= 0:
                    count = 10
                elif count > 50:  # Limit max logs to prevent spam
                    count = 50
            except ValueError:
                await update.message.reply_text(f"❌ Invalid number: {context.args[0]}. Please provide a valid number.")
                # Update the last message time after sending the response
                self._update_last_message_time()
                return
        
        logs = self.get_detection_logs(count)
        
        if not logs:
            await update.message.reply_text("📝 No detection logs available.")
            # Update the last message time after sending the response
            self._update_last_message_time()
            return
        
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
        
        await update.message.reply_text(response)
        # Update the last message time after sending the response
        self._update_last_message_time()
    
    async def _unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unknown commands."""
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning("Command response dropped due to rate limit. User sent unknown command")
            await update.message.reply_text("⏳ Please wait before sending another command. Rate limit exceeded.")
            return
            
        await update.message.reply_text("❌ Unknown command. Use /help to see available commands.")
        # Update the last message time after sending the response
        self._update_last_message_time()
    
    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Log errors caused by updates."""
        self.logger.error(f"Update {update} caused error {context.error}", exc_info=True)

    def _start_queue_processor(self):
        """
        Start the message queue processor thread.
        """
        self.queue_processor_running = True
        self.queue_processor_thread = threading.Thread(target=self._process_message_queue, daemon=True)
        self.queue_processor_thread.start()

    def _process_message_queue(self):
        """
        Process messages in the queue with rate limiting.
        """
        while self.queue_processor_running:
            try:
                # Get a message from the queue with timeout
                queued_message = self.message_queue.get(timeout=1)
                
                if queued_message:
                    # Send the message with rate limiting (the send_message method handles rate limiting)
                    result = self.send_message(queued_message.chat_id, queued_message.message,
                                             queued_message.image_path, queued_message.timeout)
                    
                    # Call the callback if provided
                    if queued_message.callback:
                        queued_message.callback(result, queued_message.chat_id)
                
                self.message_queue.task_done()
                
            except queue.Empty:
                # Continue the loop if the queue is empty
                continue
            except Exception as e:
                self.logger.error(f"Error in message queue processor: {e}")
                continue


    def _load_authorized_users(self):
        """
        Load authorized users from the config file.
        
        Returns:
            set: Set of authorized user IDs
        """
        try:
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

    def get_user_by_username(self, username, timeout=30):
        """
        Get user information by username using Telegram's getChat API.
        
        Args:
            username (str): The username to look up (with or without @)
            timeout (int): Timeout for the request in seconds (default 30)
            
        Returns:
            dict: User information if found, None otherwise
        """
        try:
            # Remove @ if present
            if username.startswith('@'):
                username = username[1:]
            
            # Using the bot's get_chat method to get user information
            chat = self.bot.get_chat(f'@{username}')
            return chat.to_dict() if chat else None
        except Exception as e:
            self.logger.error(f"Error getting user by username: {e}")
            return None

    def add_user_by_username(self, username, timeout=30):
        """
        Add a user by username.
        
        Args:
            username (str): The username to add (with or without @)
            timeout (int): Timeout for the request in seconds (default 30)
            
        Returns:
            bool: True if user was added successfully, False otherwise
        """
        try:
            user_info = self.get_user_by_username(username, timeout=timeout)
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

    def send_message(self, chat_id, message, image_path=None, timeout=30):
        """
        Send a message synchronously by wrapping the async method with thread synchronization.
        This should only be used for critical messages that require immediate confirmation.
        
        Args:
            chat_id (int): The chat ID to send the message to
            message (str): The message text to send
            image_path (str, optional): Path to an image file to send with the message
            timeout (int): Timeout for the request in seconds (default 30)
            
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        # Check rate limit before sending
        if not self._check_rate_limit():
            self.logger.warning(f"Message dropped due to rate limit. Chat ID: {chat_id}, Message: {message[:50]}...")
            return False
            
        import asyncio
        import concurrent.futures
        
        try:
            def run_async():
                async def send():
                    try:
                        if image_path and Path(image_path).exists():
                            # Send message with image
                            with open(image_path, 'rb') as image_file:
                                await self.bot.send_photo(chat_id=chat_id, photo=image_file, caption=message)
                            self.logger.info(f"Message with image sent successfully to chat {chat_id}")
                            # Update the last message time after successful send
                            self._update_last_message_time()
                            return True
                        else:
                            # Send text-only message
                            await self.bot.send_message(chat_id=chat_id, text=message)
                            self.logger.info(f"Text message sent successfully to chat {chat_id}")
                            # Update the last message time after successful send
                            self._update_last_message_time()
                            return True
                    except TelegramError as e:
                        self.logger.error(f"Failed to send message to chat {chat_id}: {e}")
                        return False
                
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(send())
                finally:
                    loop.close()
                return result
            
            # Run in a separate thread to avoid nested event loop issues
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                result = future.result(timeout=timeout+5)
            
            return result
        except Exception as e:
            self.logger.error(f"Unexpected error sending message to chat {chat_id}: {e}")
            return False

    
    def send_message_async(self, chat_id, message, image_path=None, timeout=30, callback=None):
        """
        Send a message asynchronously with rate limiting.
        
        Args:
            chat_id (int): The chat ID to send the message to
            message (str): The message text to send
            image_path (str, optional): Path to an image file to send with the message
            timeout (int): Timeout for the request in seconds (default 30)
            callback (callable, optional): Function to call with the result (success, result)
            
        Returns:
            bool: True if message was queued for sending, False if dropped due to rate limit
        """
        # Check rate limit before queuing
        if not self._check_rate_limit():
            self.logger.warning(f"Message dropped due to rate limit. Chat ID: {chat_id}, Message: {message[:50]}...")
            return False
            
        # Create a queued message
        queued_message = QueuedMessage(
            chat_id=chat_id,
            message=message,
            image_path=image_path,
            timeout=timeout,
            timestamp=time.time(),
            callback=callback
        )
        
        # Add the message to the queue
        self.message_queue.put(queued_message)
        self.logger.debug(f"Message queued for chat {chat_id}. Queue size: {self.message_queue.qsize()}")
        
        return True

    def _send_text_message(self, chat_id, message, timeout=30):
        """
        Send a text-only message to a Telegram chat with rate limiting.
        
        Args:
            chat_id (int): The chat ID to send the message to
            message (str): The message text to send
            timeout (int): Timeout for the request in seconds (default 30)
            
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        # Check rate limit before sending
        if not self._check_rate_limit():
            self.logger.warning(f"Message dropped due to rate limit. Chat ID: {chat_id}, Message: {message[:50]}...")
            return False
            
        import asyncio
        import concurrent.futures
        
        try:
            def run_async():
                async def send():
                    try:
                        await self.bot.send_message(chat_id=chat_id, text=message)
                        self.logger.info(f"Text message sent successfully to chat {chat_id}")
                        # Update the last message time after successful send
                        self._update_last_message_time()
                        return True
                    except TelegramError as e:
                        self.logger.error(f"Failed to send text message to chat {chat_id}: {e}")
                        return False
                
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(send())
                finally:
                    loop.close()
                return result
            
            # Run in a separate thread to avoid nested event loop issues
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                result = future.result(timeout=timeout+5)
            
            return result
        except Exception as e:
            self.logger.error(f"Unexpected error sending text message to chat {chat_id}: {e}")
            return False


    async def get_me(self, timeout=30):
        """
        Get information about the bot.
        
        Args:
            timeout (int): Timeout parameter kept for compatibility (not used by underlying method)
        
        Returns:
            dict: Bot information if successful, None otherwise
        """
        try:
            bot_info = await self.bot.get_me()
            return bot_info.to_dict() if bot_info else None
        except Exception as e:
            self.logger.error(f"Error getting bot info: {e}")
            return None

    def handle_command(self, user_id, command, args):
        """
        Handle incoming commands from users. (This method is maintained for compatibility)
        
        Args:
            user_id (int): The ID of the user who sent the command
            command (str): The command text (e.g., 'start', 'adduser')
            args (list): List of arguments for the command
            
        Returns:
            str: Response message to send back to the user
        """
        # Check rate limit before processing any command response
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: {command}")
            return "⏳ Please wait before sending another command. Rate limit exceeded."
            
        # Check if user is authorized for admin commands
        is_authorized = self.is_authorized_user(user_id)
        
        if command == 'start':
            response = self._handle_start_command(user_id)
        elif command == 'help':
            response = self._handle_help_command(user_id)
        elif command == 'adduser':
            if is_authorized:
                response = self._handle_adduser_command(user_id, args)
            else:
                response = "❌ You are not authorized to use this command."
        elif command == 'showlogs':
            if is_authorized:
                response = self._handle_showlogs_command(user_id, args)
            else:
                response = "❌ You are not authorized to use this command."
        else:
            response = "❌ Unknown command. Use /help to see available commands."
            
        # Update the last message time after preparing response
        self._update_last_message_time()
        return response

    def _handle_start_command(self, user_id):
        """
        Handle the /start command. (This method is maintained for compatibility)
        
        Args:
            user_id (int): The ID of the user who sent the command
            
        Returns:
            str: Response message
        """
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: start")
            return "⏳ Please wait before sending another command. Rate limit exceeded."
            
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
            
        # Update the last message time after preparing response
        self._update_last_message_time()
        return welcome_msg

    def _handle_help_command(self, user_id):
        """
        Handle the /help command. (This method is maintained for compatibility)
        
        Args:
            user_id (int): The ID of the user who sent the command
            
        Returns:
            str: Response message
        """
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: help")
            return "⏳ Please wait before sending another command. Rate limit exceeded."
            
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
        
        # Update the last message time after preparing response
        self._update_last_message_time()
        return help_msg

    def _handle_adduser_command(self, user_id, args):
        """
        Handle the /adduser command. (This method is maintained for compatibility)
        
        Args:
            user_id (int): The ID of the user who sent the command
            args (list): List of arguments for the command
            
        Returns:
            str: Response message
        """
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: adduser")
            return "⏳ Please wait before sending another command. Rate limit exceeded."
            
        if not args:
            response = "❌ Usage: /adduser <username|user_id>\nExample: /adduser @username or /adduser 123456789"
            # Update the last message time after preparing response
            self._update_last_message_time()
            return response
        
        target = args[0]
        
        # Check if it's a username (starts with @) or a numeric user ID
        if target.startswith('@') or target.lstrip('-').isdigit():
            if target.startswith('@'):
                # Add user by username
                success = self.add_user_by_username(target, timeout=30)
                if success:
                    response = f"✅ Successfully added user {target} to authorized list."
                else:
                    response = f"❌ Failed to add user {target}. Make sure the username exists and is correct."
            else:
                # Add user by ID
                try:
                    user_id_to_add = int(target)
                    success = self.add_user(user_id_to_add)
                    if success:
                        response = f"✅ Successfully added user ID {user_id_to_add} to authorized list."
                    else:
                        response = f"❌ Failed to add user ID {user_id_to_add}."
                except ValueError:
                    response = f"❌ Invalid user ID: {target}. User ID must be a number."
        else:
            response = f"❌ Invalid format: {target}. Use @username or numeric user ID."
            
        # Update the last message time after preparing response
        self._update_last_message_time()
        return response

    def _handle_showlogs_command(self, user_id, args):
        """
        Handle the /showlogs command. (This method is maintained for compatibility)
        
        Args:
            user_id (int): The ID of the user who sent the command
            args (list): List of arguments for the command
            
        Returns:
            str: Response message
        """
        # Check rate limit before responding
        if not self._check_rate_limit():
            self.logger.warning(f"Command response dropped due to rate limit. User ID: {user_id}, Command: showlogs")
            return "⏳ Please wait before sending another command. Rate limit exceeded."
            
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
        
        # Update the last message time after preparing response
        self._update_last_message_time()
        return response

    def start_polling(self):
        """
        Start the long polling loop to continuously receive commands.
        This method runs indefinitely until interrupted.
        """
        self.logger.info("Starting Telegram bot polling...")
        
        try:
            # Run the application with polling - this handles the entire lifecycle
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        except KeyboardInterrupt:
            self.logger.info("Polling interrupted by user")
        except Exception as e:
            self.logger.error(f"Critical error in polling: {e}")
        finally:
            self.logger.info("Telegram bot polling stopped")

    def cleanup(self):
        """
        Clean up resources used by the Telegram bot.
        """
        self.logger.info("Cleaning up Telegram bot resources...")
        
        # Stop the queue processor
        self.queue_processor_running = False
        
        # Wait for the queue processor thread to finish
        if self.queue_processor_thread and self.queue_processor_thread.is_alive():
            self.queue_processor_thread.join(timeout=2)  # Wait up to 2 seconds
        
        self.logger.info("Telegram bot cleanup complete")

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
