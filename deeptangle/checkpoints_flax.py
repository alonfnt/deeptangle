"""
Checkpoint management for Flax models.

This module provides utilities for saving and loading Flax model checkpoints
in a format compatible with HuggingFace Hub.
"""
from pathlib import Path
from typing import Any
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints as flax_checkpoints
from flax.training import train_state


def save_checkpoint(
    checkpoint_dir: str,
    state: train_state.TrainState,
    step: int,
    keep: int = 5,
    overwrite: bool = False
) -> None:
    """
    Save a Flax training state checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoints.
        state: Training state to save.
        step: Current training step.
        keep: Number of checkpoints to keep.
        overwrite: Whether to overwrite existing checkpoint.
    """
    checkpoint_path = Path(checkpoint_dir).absolute()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    flax_checkpoints.save_checkpoint(
        ckpt_dir=str(checkpoint_path),
        target=state,
        step=step,
        keep=keep,
        overwrite=overwrite
    )


def restore_checkpoint(
    checkpoint_dir: str,
    state: train_state.TrainState,
    step: int = None
) -> train_state.TrainState:
    """
    Restore a Flax training state checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints.
        state: Template training state for structure.
        step: Specific step to restore (None for latest).
        
    Returns:
        Restored training state.
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
    
    return flax_checkpoints.restore_checkpoint(
        ckpt_dir=str(checkpoint_path),
        target=state,
        step=step
    )


def save_model_for_inference(
    output_dir: str,
    params: Any,
    batch_stats: Any,
    pca_matrix: jnp.ndarray,
    metadata: dict
) -> None:
    """
    Save model in a format suitable for inference and HuggingFace Hub.
    
    Args:
        output_dir: Directory to save the model.
        params: Model parameters.
        batch_stats: Batch normalization statistics.
        pca_matrix: PCA transformation matrix.
        metadata: Model metadata (architecture, hyperparameters, etc.).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    params_path = output_path / "params.npy"
    with params_path.open('wb') as f:
        # Flatten and save all arrays
        flat_params, tree_def = jax.tree_util.tree_flatten(params)
        for arr in flat_params:
            jnp.save(f, arr, allow_pickle=False)
    
    # Save tree structure
    tree_path = output_path / "params_tree.pkl"
    import pickle
    with tree_path.open('wb') as f:
        pickle.dump(tree_def, f)
    
    # Save batch stats if available
    if batch_stats:
        batch_stats_path = output_path / "batch_stats.npy"
        with batch_stats_path.open('wb') as f:
            flat_stats, stats_tree_def = jax.tree_util.tree_flatten(batch_stats)
            for arr in flat_stats:
                jnp.save(f, arr, allow_pickle=False)
        
        stats_tree_path = output_path / "batch_stats_tree.pkl"
        with stats_tree_path.open('wb') as f:
            pickle.dump(stats_tree_def, f)
    
    # Save PCA matrix
    pca_path = output_path / "pca_matrix.npy"
    np.save(pca_path, np.array(pca_matrix))
    
    # Save metadata
    metadata_path = output_path / "model_metadata.json"
    with metadata_path.open('w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {output_dir}")


def load_model_for_inference(model_dir: str) -> tuple:
    """
    Load model from a saved directory for inference.
    
    Args:
        model_dir: Directory containing the saved model.
        
    Returns:
        Tuple of (params, batch_stats, pca_matrix, metadata)
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory {model_dir} not found")
    
    # Load parameters
    params_path = model_path / "params.npy"
    tree_path = model_path / "params_tree.pkl"
    
    import pickle
    with tree_path.open('rb') as f:
        tree_def = pickle.load(f)
    
    with params_path.open('rb') as f:
        num_arrays = len(tree_def.children())
        flat_params = [jnp.load(f) for _ in range(num_arrays)]
    
    params = jax.tree_util.tree_unflatten(tree_def, flat_params)
    
    # Load batch stats if available
    batch_stats = None
    batch_stats_path = model_path / "batch_stats.npy"
    if batch_stats_path.exists():
        stats_tree_path = model_path / "batch_stats_tree.pkl"
        with stats_tree_path.open('rb') as f:
            stats_tree_def = pickle.load(f)
        
        with batch_stats_path.open('rb') as f:
            num_stats = len(stats_tree_def.children())
            flat_stats = [jnp.load(f) for _ in range(num_stats)]
        
        batch_stats = jax.tree_util.tree_unflatten(stats_tree_def, flat_stats)
    
    # Load PCA matrix
    pca_path = model_path / "pca_matrix.npy"
    pca_matrix = jnp.array(np.load(pca_path))
    
    # Load metadata
    metadata_path = model_path / "model_metadata.json"
    with metadata_path.open('r') as f:
        metadata = json.load(f)
    
    return params, batch_stats, pca_matrix, metadata
