"""Register and export the paired classifier for timm model creation."""

from .baseline import BaselineClassifier, baseline_classifier

__all__ = ["BaselineClassifier", "baseline_classifier"]
