"""Collate paired-feature batches and prefetch them to the target device."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import suppress
from functools import partial
from numbers import Number

import numpy as np
import torch
import torch.utils.data
from torch.utils.data._utils.collate import default_collate

_logger = logging.getLogger(__name__)

numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def numpy_to_torch_dtype(np_dtype):
    """Map a NumPy scalar type or dtype to the corresponding Torch dtype."""
    try:
        return numpy_to_torch_dtype_dict[np_dtype]
    except KeyError:
        return numpy_to_torch_dtype_dict[np_dtype.type]


def fast_collate(batch):  # noqa: PLR0911
    """Stack nested samples while preserving their modality structure.

    Arrays and tensors at each position must have matching shapes. No padding
    is added here. The dataset supplies fixed-length training bags; evaluation
    keeps variable-length bags and uses batch size one.

    Args:
        batch: Nonempty sequence of samples with matching nested structure.
            Leaves may be tensors, NumPy arrays, scalar values, or strings.

    Returns:
        The same mapping, list, or tuple structure, with arrays and tensors
        stacked along a new batch dimension. String identifiers remain lists.

    Raises:
        TypeError: If a leaf has an unsupported type.
    """
    batch_size = len(batch)
    if isinstance(batch[0], tuple | list):
        is_tuple = isinstance(batch[0], tuple)
        num_fields = len(batch[0])
        if num_fields and isinstance(batch[0][0], np.ndarray):
            output_tensors: list[torch.Tensor] = []
            for j in range(num_fields):
                output_tensors.append(  # noqa: PERF401
                    torch.empty(
                        (batch_size, *batch[0][j].shape),
                        dtype=numpy_to_torch_dtype(batch[0][j].dtype),
                    )
                )
            for j in range(num_fields):
                out_j = output_tensors[j]
                for i in range(batch_size):
                    out_j[i].copy_(torch.from_numpy(batch[i][j]))
            return tuple(output_tensors) if is_tuple else output_tensors
        if num_fields and isinstance(batch[0][0], torch.Tensor):
            output_tensors: list[torch.Tensor] = []  # type: ignore[no-redef]
            for j in range(num_fields):
                output_tensors.append(
                    torch.empty(
                        (batch_size, *batch[0][j].shape), dtype=batch[0][j].dtype
                    )
                )
            for j in range(num_fields):
                out_j = output_tensors[j]
                for i in range(batch_size):
                    out_j[i].copy_(batch[i][j])
            return tuple(output_tensors) if is_tuple else output_tensors
        output = [fast_collate([item[i] for item in batch]) for i in range(num_fields)]
        return tuple(output) if is_tuple else output
    if isinstance(batch[0], Mapping):
        return {key: fast_collate([item[key] for item in batch]) for key in batch[0]}
    if isinstance(batch[0], np.ndarray):
        output_tensor = torch.empty(
            (batch_size, *batch[0].shape),
            dtype=numpy_to_torch_dtype(batch[0].dtype),
        )
        for i in range(batch_size):
            output_tensor[i].copy_(torch.from_numpy(batch[i]))
        return output_tensor
    if isinstance(batch[0], torch.Tensor):
        output_tensor = torch.empty((batch_size, *batch[0].shape), dtype=batch[0].dtype)
        for i in range(batch_size):
            output_tensor[i].copy_(batch[i])
        return output_tensor
    if isinstance(batch[0], str | bytes | Number | np.generic):
        return default_collate(batch)
    raise TypeError(f"Unsupported batch type: {type(batch[0])}")


class PrefetchLoader:
    """Yield device-ready batches with overlapping CUDA transfers.

    CPU batches follow the same recursive traversal without a CUDA stream.

    Args:
        loader: DataLoader yielding nested tensors and patient identifiers.
        device: Target device for tensor data.
        cast_dtype: Optional dtype for floating features. Integer labels keep
            their dtype, and non-tensor metadata is unchanged.
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda"),  # noqa: B008
        cast_dtype: torch.dtype | None = None,
    ):
        self.loader = loader
        self.device = device
        self.is_cuda = device.type == "cuda" and torch.cuda.is_available()
        self.cast_dtype = cast_dtype

    def _move_batch_to_device(self, batch):
        """Move nested tensors while preserving label dtypes and metadata."""
        if isinstance(batch, torch.Tensor):
            batch = batch.to(
                device=self.device,
                dtype=self.cast_dtype
                if self.cast_dtype is not None and batch.is_floating_point()
                else None,
                non_blocking=True,
            )
        elif isinstance(batch, Mapping):
            batch = {
                key: self._move_batch_to_device(value) for key, value in batch.items()
            }
        elif isinstance(batch, list):
            batch = [self._move_batch_to_device(value) for value in batch]
        elif isinstance(batch, tuple):
            batch = tuple(self._move_batch_to_device(value) for value in batch)
        return batch

    def _record_stream(self, batch, stream):
        """Keep nested tensor storage in use until the consumer stream finishes."""
        if isinstance(batch, torch.Tensor):
            batch.record_stream(stream)
        elif isinstance(batch, Mapping):
            for value in batch.values():
                self._record_stream(value, stream)
        elif isinstance(batch, (list, tuple)):
            for value in batch:
                self._record_stream(value, stream)

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream(device=self.device)
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress  # type: ignore[assignment]

        for next_batch in self.loader:
            with stream_context():
                next_batch = self._move_batch_to_device(next_batch)  # noqa: PLW2901

            if not first:
                # On CUDA, the next copy overlaps computation on this batch.
                yield batch  # type: ignore[has-type]  # noqa: F821
            else:
                first = False

            if stream is not None:
                if self.is_cuda:
                    consumer_stream = torch.cuda.current_stream(device=self.device)
                    # The consumer must wait for this batch's queued copies.
                    consumer_stream.wait_stream(stream)
                    # Keep transferred storage alive while the consuming stream uses it.
                    self._record_stream(next_batch, consumer_stream)

            batch = next_batch

        if not first:
            yield batch

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
