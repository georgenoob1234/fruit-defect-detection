import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a YAML configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Raises:
        ValueError: If required configuration sections are missing or invalid
    """
    # Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}
    
    # Validate telegram cooldown configuration if present
    if 'telegram' in config and 'cooldown' in config['telegram']:
        cooldown = config['telegram']['cooldown']
        
        # Validate required cooldown parameters
        required_params = ['duration', 'global_cooldown', 'per_chat_cooldown',
                          'per_chat_duration', 'message_deduplication',
                          'deduplication_window', 'max_queue_size']
        
        for param in required_params:
            if param not in cooldown:
                raise ValueError(f"Missing required cooldown parameter: {param}")
            
        # Validate parameter types and values
        if not isinstance(cooldown['duration'], (int, float)) or cooldown['duration'] <= 0:
            raise ValueError("cooldown.duration must be a positive number")
            
        if not isinstance(cooldown['per_chat_duration'], (int, float)) or cooldown['per_chat_duration'] <= 0:
            raise ValueError("cooldown.per_chat_duration must be a positive number")
            
        if not isinstance(cooldown['deduplication_window'], (int, float)) or cooldown['deduplication_window'] <= 0:
            raise ValueError("cooldown.deduplication_window must be a positive number")
            
        if not isinstance(cooldown['max_queue_size'], int) or cooldown['max_queue_size'] <= 0:
            raise ValueError("cooldown.max_queue_size must be a positive integer")
            
        for bool_param in ['global_cooldown', 'per_chat_cooldown', 'message_deduplication']:
            if not isinstance(cooldown[bool_param], bool):
                raise ValueError(f"cooldown.{bool_param} must be a boolean value")
    
    # Validate telegram rate limiting configuration if present
    if 'telegram' in config and 'rate_limiting' in config['telegram']:
        rate_limiting = config['telegram']['rate_limiting']
        
        # Validate parameter types and values
        if 'min_interval' in rate_limiting:
            if not isinstance(rate_limiting['min_interval'], (int, float)) or rate_limiting['min_interval'] < 0:
                raise ValueError("rate_limiting.min_interval must be a non-negative number")
                
        if 'max_retries' in rate_limiting:
            if not isinstance(rate_limiting['max_retries'], int) or rate_limiting['max_retries'] < 0:
                raise ValueError("rate_limiting.max_retries must be a non-negative integer")
                
        if 'base_delay' in rate_limiting:
            if not isinstance(rate_limiting['base_delay'], (int, float)) or rate_limiting['base_delay'] <= 0:
                raise ValueError("rate_limiting.base_delay must be a positive number")
                
        if 'max_delay' in rate_limiting:
            if not isinstance(rate_limiting['max_delay'], (int, float)) or rate_limiting['max_delay'] <= 0:
                raise ValueError("rate_limiting.max_delay must be a positive number")
    
    return config