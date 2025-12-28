import os
import h5py
import torch
import numpy as np
from typing import List, Dict, Union


class EmbeddingStore:
    """
    Reusable class for saving and loading embeddings by sample IDs.

    Features:
    - Save embeddings (torch.Tensor / np.ndarray) to HDF5
    - Load single or multiple embeddings by ID
    - Supports efficient subsampling without loading entire dataset
    - Optional compression via gzip

    Usage:
        with EmbeddingStore("embeddings.h5") as store:
            store.save({"id1": vec1, "id2": vec2})
            subset = store.load(["id1", "id2"])
            all_ids = store.keys()
    """

    def __init__(self, path: str, mode: str = "a"):
        """
        Parameters
        ----------
        path : str
            Path to HDF5 file.
        mode : {"a", "r", "w"}
            File mode: append, read-only, write (overwrite)
        """
        self.path = path
        self.mode = mode
        self._file = None

    def __enter__(self):
        self._file = h5py.File(self.path, self.mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file is not None:
            self._file.close()
            self._file = None

    # -----------------------------------------------------------------
    # Keys
    # -----------------------------------------------------------------
    def keys(self) -> List[str]:
        """Return all IDs stored in this HDF5 embedding store."""
        if self._file is None:
            raise RuntimeError("Use 'with' context to open the store")
        return list(self._file.keys())


    # -----------------------------------------------------------------
    # Saving embeddings
    # -----------------------------------------------------------------
    def save(self, embeddings: Dict[str, Union[torch.Tensor, np.ndarray]]):
        """
        Save multiple embeddings.

        Parameters
        ----------
        embeddings : dict
            Mapping: sample_id -> embedding (tensor or ndarray)
        """
        if self._file is None:
            raise RuntimeError("Use 'with' context to open the store")

        for sid, vec in embeddings.items():
            # Convert torch → numpy
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().cpu().numpy()

            # Force floating point (CRITICAL)
            if not np.issubdtype(vec.dtype, np.floating):
                vec = vec.astype(np.float32)

            # Sanity check
            if not np.isfinite(vec).all():
                raise ValueError(f"Embedding {sid} contains NaN or inf")

            if vec.ndim != 1:
                raise ValueError(f"Embedding {sid} must be 1D, got shape {vec.shape}")

            # Overwrite safely
            if sid in self._file:
                del self._file[sid]

            self._file.create_dataset(
                sid,
                data=vec,
                dtype=vec.dtype,
                compression="gzip"
            )
    
    # -----------------------------------------------------------------
    # Saving embeddings OLD
    # -----------------------------------------------------------------
    def save_old(self, embeddings: Dict[str, Union[torch.Tensor, np.ndarray]]):
        """
        Save multiple embeddings.

        Parameters
        ----------
        embeddings : dict
            Mapping: sample_id -> embedding (tensor or ndarray)
        """
        if self._file is None:
            raise RuntimeError("Use 'with' context to open the store")

        for sid, vec in embeddings.items():
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().cpu().numpy()
            if sid in self._file:
                del self._file[sid]  # overwrite
            self._file.create_dataset(sid, data=vec, compression="gzip")

    # -----------------------------------------------------------------
    # Loading embeddings
    # -----------------------------------------------------------------
    def load(self, ids: Union[List[str], str]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Load embeddings for the given IDs.

        Parameters
        ----------
        ids : list[str] or str
            One or multiple IDs to load.

        Returns
        -------
        dict or np.ndarray:
            If a list of IDs is provided, returns dict {id: embedding}.
            If a single ID is provided, returns np.ndarray.
        """
        if self._file is None:
            raise RuntimeError("Use 'with' context to open the store")

        if isinstance(ids, str):
            # single ID
            if ids in self._file:
                return self._file[ids][()]
            else:
                raise KeyError(f"ID {ids} not found in store")
        else:
            # list of IDs
            result = {}
            for sid in ids:
                if sid in self._file:
                    result[sid] = self._file[sid][()]
                else:
                    raise KeyError(f"ID {sid} not found in store")
            return result

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------
    def all_ids(self) -> List[str]:
        """Return a list of all IDs in the store (alias for keys)."""
        return self.keys()
