"""Read-only shard mappings with a bounded cache of open files."""

import contextlib
import os
from collections import OrderedDict

import numpy as np


class ShardMMapLRUCache:
    """Cache shard mappings with least-recently-used eviction.

    The cache owns each returned mapping. Callers must copy needed data before
    another lookup can evict and close its backing mapping.
    """

    def __init__(
        self,
        shards_dir: str,
        on_disk_dtype: np.dtype,
        max_cache_shards: int,
    ):
        """Configure the shard locations and cache capacity.

        Args:
            shards_dir: Directory containing files named shard_0000.bin, etc.
            on_disk_dtype: NumPy dtype used to interpret the stored bytes.
            max_cache_shards: Maximum number of mappings kept open.
        """
        self.shards_dir = shards_dir
        self.on_disk_dtype = on_disk_dtype
        self.itemsize = np.dtype(on_disk_dtype).itemsize
        self.max_cache_shards = max_cache_shards
        self.lru: OrderedDict[int, np.memmap] = OrderedDict()

    def open_memmap(self, shard_id: int) -> np.memmap:
        """Open one shard as a flat, read-only array of scalar elements."""
        shard_file = os.path.join(self.shards_dir, f"shard_{shard_id:04d}.bin")
        if not os.path.exists(shard_file):
            raise FileNotFoundError(f"Shard file not found: {shard_file}")
        num_bytes = os.path.getsize(shard_file)
        if num_bytes % self.itemsize != 0:
            raise ValueError("Corrupt shard: size not divisible by dtype itemsize")
        num_elems = num_bytes // self.itemsize
        return np.memmap(
            shard_file, mode="r", dtype=self.on_disk_dtype, shape=(num_elems,)
        )

    def close_memmap(self, mm: np.memmap) -> None:
        # NumPy exposes the backing mapping through _mmap rather than close().
        m = getattr(mm, "_mmap", None)
        if m is not None:
            with contextlib.suppress(Exception):
                m.close()

    def get(self, shard_id: int) -> np.memmap:
        """Return a cached mapping, opening the shard on a cache miss."""
        mm = self.lru.get(shard_id, None)
        if mm is not None:
            self.lru.move_to_end(shard_id)
            return mm

        mm = self.open_memmap(shard_id)
        self.lru[shard_id] = mm
        self.lru.move_to_end(shard_id, last=True)

        while len(self.lru) > self.max_cache_shards:
            _, evicted_mm = self.lru.popitem(last=False)
            self.close_memmap(evicted_mm)

        return mm

    def clear(self) -> None:
        for mm in self.lru.values():
            self.close_memmap(mm)
        self.lru.clear()
