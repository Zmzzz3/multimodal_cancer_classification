"""Translate index dtype labels to NumPy storage and PyTorch tensor dtypes."""

import numpy as np
import torch

_NP_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "uint8": np.uint8,
    "int8": np.int8,
}
_TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
_NP_TO_TORCH_DTYPE_MAP = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
}
