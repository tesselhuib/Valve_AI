"""This module contains the function get_transform() to get the transformation
pipeline used in the project.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torchvision import transforms
from config import INPUT_SIZE

def get_transform():
    """Returns the transformation pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(INPUT_SIZE),
        transforms.Grayscale()
    ])
