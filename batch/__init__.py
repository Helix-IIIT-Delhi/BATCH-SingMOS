"""BATCH: Bhattacharyya Distance Fusion for SingMOS Prediction"""

__version__ = "1.0.0"
__author__ = "Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, and others"

from .fusion import BATCHFusion
from .distance import bhattacharyya_distance

__all__ = ["BATCHFusion", "bhattacharyya_distance"]
