"""
Src module initialization for the Fruit Defect Detection System.

This module contains the main components for fruit detection and defect segmentation.
"""
from . import utils
from . import detection
from . import api
from . import camera
from . import telegram
from . import gui

__all__ = [
    'utils',
    'detection',
    'api',
    'camera',
    'telegram',
    'gui'
]