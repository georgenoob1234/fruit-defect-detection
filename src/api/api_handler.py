"""
API Handler for the Fruit Defect Detection System.

This module handles sending detection data and images to a web API.
"""

import logging
import requests
import json
from pathlib import Path


class APIHandler:
    """
    A class to handle API requests for sending detection data and images.
    """
    
    def __init__(self, base_url, endpoint, timeout=10):
        """
        Initialize the API handler with the provided configuration.
        
        Args:
            base_url (str): The base URL of the API
            endpoint (str): The endpoint to send detection data to
            timeout (int): Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.endpoint = endpoint.lstrip('/')
        self.url = f"{self.base_url}/{self.endpoint}"
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def send_detection(self, detection_data, image_path=None):
        """
        Send detection data to the API, optionally with an image.
        
        Args:
            detection_data (dict): Detection information to send
            image_path (str, optional): Path to an image file to upload with the detection
            
        Returns:
            bool: True if the data was sent successfully, False otherwise
        """
        try:
            if image_path and Path(image_path).exists():
                # Send with image as multipart/form-data
                with open(image_path, 'rb') as image_file:
                    files = {
                        'image': (Path(image_path).name, image_file, 'image/jpeg')
                    }
                    
                    # Add detection data as JSON in the form
                    data = {
                        'detection_data': json.dumps(detection_data)
                    }
                    
                    response = requests.post(
                        self.url,
                        files=files,
                        data=data,
                        timeout=self.timeout
                    )
            else:
                # Send detection data as JSON
                headers = {
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(
                    self.url,
                    json=detection_data,
                    headers=headers,
                    timeout=self.timeout
                )
            
            if response.status_code in [200, 201]:
                self.logger.info(f"Detection data sent successfully to {self.url}")
                return True
            else:
                self.logger.error(f"Failed to send detection data: {response.status_code} - {response.text}")
                return False
                
        except FileNotFoundError:
            self.logger.error(f"Image file not found: {image_path}")
            # Send text-only data instead
            return self._send_detection_data_only(detection_data)
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout after {self.timeout} seconds")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error when sending detection data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error when sending detection data: {e}")
            return False

    def _send_detection_data_only(self, detection_data):
        """
        Send detection data only (without image) to the API.
        
        Args:
            detection_data (dict): Detection information to send
            
        Returns:
            bool: True if the data was sent successfully, False otherwise
        """
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.url,
                json=detection_data,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code in [200, 201]:
                self.logger.info(f"Detection data sent successfully to {self.url} (without image)")
                return True
            else:
                self.logger.error(f"Failed to send detection data: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout after {self.timeout} seconds")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error when sending detection data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error when sending detection data: {e}")
            return False

    def test_connection(self):
        """
        Test the connection to the API endpoint.
        
        Returns:
            bool: True if the connection is successful, False otherwise
        """
        try:
            response = requests.get(self.base_url, timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def check_endpoint(self):
        """
        Check if the specific endpoint is accessible.
        
        Returns:
            bool: True if the endpoint is accessible, False otherwise
        """
        try:
            # Try to send a minimal request to check if the endpoint exists
            response = requests.options(self.url, timeout=self.timeout)
            # If we get any response (even 405), it means the endpoint exists
            return response.status_code != 404
        except Exception as e:
            self.logger.error(f"Endpoint check failed: {e}")
            return False