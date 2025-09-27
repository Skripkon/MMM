"""
Multi-Modal Models (MMM) - A comprehensive framework for multi-modal machine learning.

This package provides tools and models for working with multiple data modalities
including text, images, audio, and video.
"""

__version__ = "0.1.0"
__author__ = "HSE AI Lab"
__email__ = "ai-lab@hse.ru"

from . import models, data, training, evaluation

__all__ = ["models", "data", "training", "evaluation"]