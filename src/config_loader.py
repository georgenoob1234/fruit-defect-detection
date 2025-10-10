import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load and return the YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if config is None:
        # Return an empty dict if the file is empty
        return {}
    
    return config