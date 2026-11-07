"""Register and export multimodal and unimodal classifiers for timm."""

from .baseline import (
    BaselineClassifier, RNAOnlyClassifier, WSIOnlyClassifier,
    baseline_classifier, rna_only_classifier, wsi_only_classifier,
)

__all__ = [
    "BaselineClassifier", "RNAOnlyClassifier", "WSIOnlyClassifier",
    "baseline_classifier", "rna_only_classifier", "wsi_only_classifier",
]
