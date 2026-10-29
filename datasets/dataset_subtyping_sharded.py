"""Paired WSI and RNA loading from indexed feature shards."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets._cache import ShardMMapLRUCache
from datasets._dtype_map import _NP_DTYPE_MAP, _NP_TO_TORCH_DTYPE_MAP


class SlideOmicsShardedSubtypingDataset(Dataset):
    """Load one patient split as paired float32 RNA and WSI tensors."""

    def __init__(self, data_dir, split="train", slide_seq_len=None):
        """Load the supplied split manifest and validate its feature index.

        Args:
            data_dir: Directory containing the index, split CSV and feature shards.
            split: One of "train", "val" or "test".
            slide_seq_len: Optional bag length for training batches. None keeps
                all patches; shorter bags are filled by sampling existing patches.

        Raises:
            ValueError: The index, split membership or label contract is invalid.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        if slide_seq_len is not None and (
            not isinstance(slide_seq_len, int) or isinstance(slide_seq_len, bool) or slide_seq_len <= 0
        ):
            raise ValueError("slide_seq_len must be a positive integer")
        self.slide_seq_len = slide_seq_len
        self.classes = ("LUAD", "LUSC")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        split_frame = pd.read_csv(
            self.data_dir / "splits_0.csv", dtype=str, keep_default_na=False
        )
        if list(split_frame.columns) != ["train", "val", "test"]:
            raise ValueError("splits_0.csv must have train,val,test columns")
        split_ids = {}
        all_ids = set()
        for name in ("train", "val", "test"):
            ids = [value for value in split_frame[name] if value]
            if not ids or len(ids) != len(set(ids)) or all_ids.intersection(ids):
                raise ValueError("Splits must be nonempty, unique, and disjoint")
            if not all(re.fullmatch(r"P\d{4}", value) for value in ids):
                raise ValueError("Split identifiers must be anonymous patient IDs")
            split_ids[name] = ids
            all_ids.update(ids)

        master_index_df = pd.read_parquet(self.data_dir / "master_index.parquet")
        expected_columns = {
            "patient_id", "slide_id", "slide_shard_id", "slide_offset",
            "slide_len", "slide_dim", "slide_dtype", "gene_expression_offset",
            "gene_expression_len", "gene_expression_dtype",
        }
        if set(master_index_df.columns) != expected_columns:
            raise ValueError("Unexpected master-index fields")
        if master_index_df["patient_id"].duplicated().any() or master_index_df["slide_id"].duplicated().any():
            raise ValueError("Each patient and slide must occur once")
        if set(master_index_df["patient_id"]) != all_ids:
            raise ValueError("Master-index patients differ from the supplied splits")
        if not master_index_df["slide_id"].map(lambda value: bool(re.fullmatch(r"S\d{4}", str(value)))).all():
            raise ValueError("Slide identifiers must be anonymous")
        if master_index_df.index.tolist() != master_index_df["slide_id"].tolist():
            raise ValueError("Master-index row IDs must match slide IDs")
        if not master_index_df["slide_len"].between(1, 1024).all():
            raise ValueError("WSI bags must contain 1 to 1024 patches")
        if not (master_index_df["slide_dim"] == 768).all():
            raise ValueError("WSI feature width must be 768")
        if not (master_index_df[["slide_shard_id", "slide_offset", "gene_expression_offset"]] >= 0).all().all():
            raise ValueError("Every patient must have both WSI and RNA features")
        for prefix in ("slide", "gene_expression"):
            if set(master_index_df[f"{prefix}_dtype"]) != {"float32"}:
                raise ValueError("The supplied feature shards must use float32")
        self.labels = None
        # Test inference does not require or read labels.csv.
        if split != "test":
            labels = pd.read_csv(self.data_dir / "labels.csv", dtype=str, keep_default_na=False)
            if list(labels.columns) != ["patient_id", "label"]:
                raise ValueError("labels.csv must have patient_id,label columns")
            if labels["patient_id"].duplicated().any() or not labels["label"].isin(self.classes).all():
                raise ValueError("Labels must be unique and use LUAD or LUSC")
            if set(labels["patient_id"]) != set(split_ids["train"] + split_ids["val"]):
                raise ValueError("Labels must cover train/val patients and exclude test patients")
            label_map = dict(zip(labels["patient_id"], labels["label"].map({"LUAD": 0, "LUSC": 1}), strict=True))
            self.labels = np.asarray([label_map[patient_id] for patient_id in split_ids[split]], dtype=np.int64)

        # Preserve manifest order so predictions and identifiers stay aligned.
        master_index_df = master_index_df.set_index("patient_id", drop=False).loc[split_ids[split]]
        self.patient_ids = master_index_df["patient_id"].tolist()
        self.row_ids = master_index_df["slide_id"].tolist()
        self.used_feature_ids = self.patient_ids
        self.slide_shards = self.data_dir / "slides"
        self.slide_return_dtype = torch.float32
        self.max_cache_shards = 1
        self.mmap_cache = None
        (
            self.gene_expression_len,
            self.gene_expression_dtype,
            self.gene_expression_offsets,
            self.gene_expression_mm,
        ) = self._init_omics_feature(master_index_df, "gene_expression", self.data_dir / "rna.bin")
        self.slide_dim, self.slide_on_disk_dtype = self._extract_metadata(
            master_index_df, "slide_dim", "slide_dtype"
        )
        self.slide_shard_ids = master_index_df["slide_shard_id"].to_numpy(dtype=np.int64, copy=False)
        self.slide_offsets = master_index_df["slide_offset"].to_numpy(dtype=np.int64, copy=False)
        self.slide_lengths = master_index_df["slide_len"].to_numpy(dtype=np.int64, copy=False)
        if self.gene_expression_len <= 0:
            raise ValueError("RNA feature width must be positive")
        # Offsets count scalar elements, not bytes; WSI lengths count patch rows.
        if np.any(self.gene_expression_offsets + self.gene_expression_len > self.gene_expression_mm.size):
            raise ValueError("RNA offsets exceed the shard size")
        for shard_id in np.unique(self.slide_shard_ids):
            shard_path = self.slide_shards / f"shard_{shard_id:04d}.bin"
            size = shard_path.stat().st_size
            mask = self.slide_shard_ids == shard_id
            ends = self.slide_offsets[mask] + self.slide_lengths[mask] * self.slide_dim
            if size % 4 or np.any(ends > size // 4):
                raise ValueError("WSI offsets exceed the shard size")

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def feature_dim(self):
        return self.slide_dim

    @property
    def rna_feature_dim(self):
        return self.gene_expression_len

    def __len__(self):
        return len(self.patient_ids)

    def __getstate__(self):
        """Exclude live mappings when sending this dataset to a worker."""
        state = dict(self.__dict__)
        state["mmap_cache"] = None
        state["gene_expression_mm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Workers reopen RNA here; the WSI cache opens lazily on first access.
        self.gene_expression_mm = self._open_memmap(self.data_dir / "rna.bin", np.float32)

    def __getitem__(self, idx):
        """Return one patient's features and anonymous identifiers.

        Returns:
            A dict with x=[rna, wsi], containing float32 tensors of shapes [G]
            and [N, 768]. patient_id and row_id identify the patient and slide.
            Train/val samples include an int64 scalar label (LUAD=0, LUSC=1);
            test samples omit the label.

        Raises:
            ValueError: Either modality contains non-finite values.
        """
        rna, _ = self._load_omics_feature(
            idx, self.gene_expression_offsets, self.gene_expression_mm,
            self.gene_expression_len, self.gene_expression_dtype,
        )
        slide = self._load_slide_feature(idx)
        if not torch.isfinite(rna).all() or not torch.isfinite(slide).all():
            raise ValueError("Patient features must be finite")
        sample = {"x": [rna, slide], "patient_id": self.patient_ids[idx], "row_id": self.row_ids[idx]}
        if self.labels is not None:
            sample["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return sample

    @staticmethod
    def _load_omics_feature(idx, offsets, mm, length, dtype):
        offset = int(offsets[idx])
        if offset >= 0:
            data = mm[offset : offset + length]
            # Copy read-only mapped data into storage owned by the tensor.
            return torch.tensor(data, dtype=dtype), True
        return torch.zeros(length, dtype=dtype), False

    def _load_slide_feature(self, idx):
        shard_id = int(self.slide_shard_ids[idx])
        offset = int(self.slide_offsets[idx])
        length = int(self.slide_lengths[idx])
        mm = self.get_cache().get(shard_id)
        slide_view = mm[offset : offset + length * self.slide_dim].reshape(-1, self.slide_dim)
        target_len = self.slide_seq_len if self.slide_seq_len is not None else length
        if length > target_len:
            # Sort sampled indices for sequential access to the feature mapping.
            indices = torch.randperm(length)[:target_len].sort().values.numpy()
            feature_array = slide_view[indices]
        else:
            feature_array = slide_view
        # Copy before the cache can evict and close this shard mapping.
        tensor = torch.tensor(feature_array, dtype=self.slide_return_dtype)
        if tensor.shape[0] < target_len:
            tensor = self._pad_to_len(tensor, target_len)
        return tensor

    @staticmethod
    def _pad_to_len(
        x: torch.Tensor,
        target_len: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Robustly pads a tensor with sampled patches from itself."""
        cur, dim = x.shape
        out = x.new_empty((target_len, dim))

        if cur == 0:
            out.zero_()
            return out

        out[:cur].copy_(x)

        # Fill remainder with random patches from the existing slide
        pad_size = target_len - cur
        if pad_size > 0:
            pad_indices = torch.randint(0, cur, (pad_size,), generator=generator)
            out[cur:] = x[pad_indices]

        return out

    def get_cache(self):
        """Create the WSI mapping cache on first access in the current process."""
        if self.mmap_cache is None:
            self.mmap_cache = ShardMMapLRUCache(
                shards_dir=self.slide_shards,
                on_disk_dtype=self.slide_on_disk_dtype,
                max_cache_shards=self.max_cache_shards,
            )
        return self.mmap_cache

    def _init_omics_feature(self, df, prefix, shard_path):
        """Resolve a modality's width, tensor dtype, offsets and memory map."""
        length, on_disk_dtype = self._extract_metadata(df, f"{prefix}_len", f"{prefix}_dtype")
        offsets = df[f"{prefix}_offset"].to_numpy(dtype=np.int64, copy=False)
        mmap = self._open_memmap(shard_path, on_disk_dtype)
        torch_dtype = _NP_TO_TORCH_DTYPE_MAP[on_disk_dtype]
        return length, torch_dtype, offsets, mmap

    @staticmethod
    def _extract_metadata(df, len_col, dtype_col):
        lengths = set(df[len_col].tolist())
        lengths.discard(0)
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent {len_col}: {lengths}")
        dtypes = set(df[dtype_col].dropna().tolist())
        if len(dtypes) != 1:
            raise ValueError(f"Inconsistent {dtype_col}: {dtypes}")
        return int(lengths.pop()), _NP_DTYPE_MAP[dtypes.pop()]

    @staticmethod
    def _open_memmap(path, dtype):
        """Open a binary shard read-only without materialising its contents."""
        size = Path(path).stat().st_size
        itemsize = np.dtype(dtype).itemsize
        if size % itemsize != 0:
            raise ValueError(f"Corrupt shard {path}: size is not divisible by its dtype size")
        return np.memmap(path, mode="r", dtype=dtype, shape=(size // itemsize,))
