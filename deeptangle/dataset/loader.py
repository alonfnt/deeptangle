"""
Data loader for pre-generated synthetic datasets.

This module provides utilities to load and iterate over datasets
created by generate_dataset.py.
"""
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import jax.numpy as jnp
import jax.random as jr


class SyntheticDatasetLoader:
    """Loads pre-generated synthetic datasets from disk."""
    
    def __init__(self, dataset_dir: str, split: str = "train"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_dir: Path to the dataset directory.
            split: Either "train" or "val".
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {dataset_dir} not found")
        
        # Load metadata
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Load PCA matrix
        pca_path = self.dataset_dir / "pca_matrix.npy"
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA matrix not found at {pca_path}")
        
        self.pca_matrix = jnp.array(np.load(pca_path))
        
        # Get list of sample files
        split_dir = self.dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory {split_dir} not found")
        
        self.sample_files = sorted(list(split_dir.glob("*.npz")))
        self.num_samples = len(self.sample_files)
        
        if self.num_samples == 0:
            raise ValueError(f"No samples found in {split_dir}")
        
        print(f"Loaded {split} dataset with {self.num_samples} samples")
        print(f"PCA matrix shape: {self.pca_matrix.shape}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Tuple of (clip, label) as JAX arrays.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        sample_file = self.sample_files[idx]
        data = np.load(sample_file)
        X = jnp.array(data["X"])
        y = jnp.array(data["y"])
        return X, y
    
    def get_batch_iterator(self, batch_size: int, shuffle: bool = True, seed: int = 42):
        """
        Create an iterator that yields batches of data.
        
        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed for shuffling.
            
        Yields:
            Tuple of (batch_clips, batch_labels) as JAX arrays.
        """
        key = jr.PRNGKey(seed)
        
        while True:
            # Shuffle indices if requested
            if shuffle:
                key, shuffle_key = jr.split(key)
                indices = jr.permutation(shuffle_key, self.num_samples)
            else:
                indices = jnp.arange(self.num_samples)
            
            # Yield batches
            for start_idx in range(0, self.num_samples, batch_size):
                end_idx = min(start_idx + batch_size, self.num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_X = []
                batch_y = []
                
                for idx in batch_indices:
                    X, y = self[int(idx)]
                    batch_X.append(X)
                    batch_y.append(y)
                
                # Stack batches
                batch_X = jnp.stack(batch_X)
                # Note: batch_y might have variable sizes, so we keep it as a list
                # or pad if needed for training
                
                yield batch_X, batch_y


def load_pca_matrix(dataset_dir: str) -> jnp.ndarray:
    """Load the PCA matrix from a dataset directory."""
    pca_path = Path(dataset_dir) / "pca_matrix.npy"
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA matrix not found at {pca_path}")
    return jnp.array(np.load(pca_path))


def load_metadata(dataset_dir: str) -> Dict:
    """Load the metadata from a dataset directory."""
    metadata_path = Path(dataset_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    with open(metadata_path, "r") as f:
        return json.load(f)
