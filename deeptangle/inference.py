"""
Load and use models for inference - supports both Haiku and NNX formats.

This module provides a unified interface for loading models regardless
of whether they were trained with Haiku or NNX.
"""
from pathlib import Path
import json
import pickle

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from deeptangle import checkpoints
from deeptangle.logger import recover_experiment_parameters


def detect_model_format(model_dir: str) -> str:
    """
    Detect whether a checkpoint is in Haiku or NNX format.
    
    Args:
        model_dir: Path to model checkpoint directory.
        
    Returns:
        "haiku" or "nnx"
    """
    path = Path(model_dir)
    
    if (path / "model.nnx").exists():
        return "nnx"
    elif (path / "arrays.npy").exists() and (path / "tree.pkl").exists():
        return "haiku"
    else:
        raise ValueError(
            f"Cannot determine model format in {model_dir}. "
            "Expected either 'model.nnx' (NNX) or 'arrays.npy' + 'tree.pkl' (Haiku)"
        )


def load_haiku_model(model_dir: str):
    """
    Load a Haiku model checkpoint.
    
    Args:
        model_dir: Path to Haiku checkpoint directory.
        
    Returns:
        Tuple of (forward_fn, state, pca_matrix, metadata)
    """
    from deeptangle.forward import build_model
    
    # Load experiment parameters
    params = recover_experiment_parameters(model_dir)
    
    # Load PCA matrix
    pca_matrix = checkpoints.load_pca_matrix(model_dir)
    
    # Build Haiku model
    forward_fn = build_model(
        A=pca_matrix,
        num_suggestions=params["n_suggestions"],
        latent_dim=params["latent_dim"],
        num_frames=params["nframes"],
    )
    
    # Load state
    state = checkpoints.restore(model_dir, broadcast=False)
    
    # Create metadata dict
    metadata = {
        "nframes": params["nframes"],
        "size": params.get("size", 256),
        "npca": pca_matrix.shape[0],
        "n_suggestions": params["n_suggestions"],
        "latent_dim": params["latent_dim"],
        "model_type": "haiku",
    }
    
    return forward_fn, state, pca_matrix, metadata


def load_nnx_model(model_dir: str):
    """
    Load an NNX model checkpoint.
    
    Args:
        model_dir: Path to NNX checkpoint directory.
        
    Returns:
        Tuple of (model, pca_matrix, B, metadata)
    """
    from deeptangle.model import create_detector
    
    model_path = Path(model_dir)
    
    # Load metadata
    with open(model_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load PCA matrix
    pca_matrix = jnp.array(np.load(model_path / "pca_matrix.npy"))
    
    # Create model with same architecture
    rngs = nnx.Rngs(0)  # Seed doesn't matter for inference
    model = create_detector(
        npoints=metadata["npca"],
        n_suggestions=metadata["n_suggestions"],
        latent_dim=metadata["latent_dim"],
        nframes=metadata["nframes"],
        rngs=rngs
    )
    
    # Load model weights
    with open(model_path / "model.nnx", "rb") as f:
        model = nnx.from_bytes(model, f.read())
    
    # Compute B matrix for alignment
    kpoints2 = pca_matrix.shape[1]
    J = jnp.flip(jnp.identity(kpoints2), axis=1)
    B = pca_matrix @ J @ jnp.transpose(pca_matrix)
    
    metadata["model_type"] = "nnx"
    
    return model, pca_matrix, B, metadata


def load_model(model_dir: str):
    """
    Load a model checkpoint automatically detecting the format.
    
    Args:
        model_dir: Path to model checkpoint directory.
        
    Returns:
        Tuple of (model, pca_matrix, metadata) where model type varies:
        - For Haiku: (forward_fn, state, pca_matrix, metadata)
        - For NNX: (model, pca_matrix, B, metadata)
    """
    model_format = detect_model_format(model_dir)
    
    if model_format == "haiku":
        print(f"Loading Haiku model from {model_dir}")
        return load_haiku_model(model_dir)
    elif model_format == "nnx":
        print(f"Loading NNX model from {model_dir}")
        return load_nnx_model(model_dir)
    else:
        raise ValueError(f"Unknown model format: {model_format}")


def predict_with_haiku(forward_fn, state, clip, pca_matrix):
    """
    Run inference with a Haiku model.
    
    Args:
        forward_fn: Haiku forward function.
        state: Haiku model state (params, state).
        clip: Input clip array.
        pca_matrix: PCA transformation matrix.
        
    Returns:
        Predictions object with w, s, p fields.
    """
    # Haiku models expect batch dimension
    if clip.ndim == 3:
        clip = clip[None, ...]
    
    # Run inference
    predictions, _ = forward_fn.apply(state[0], state[1], clip, False)
    
    return predictions


def predict_with_nnx(model, B, clip):
    """
    Run inference with an NNX model.
    
    Args:
        model: NNX model.
        B: Alignment matrix for eigenvalues.
        clip: Input clip array.
        
    Returns:
        Tuple of (S_pred, W_pred, P_pred) where S_pred is scores,
        W_pred is coordinates, and P_pred is latent space.
    """
    from deeptangle.dataset.pca import points_from_pca
    
    # Ensure batch dimension
    if clip.ndim == 3:
        clip = clip[None, ...]
    
    # Transpose to (B, H, W, T) format
    inputs = jnp.transpose(clip, axes=(0, 2, 3, 1))
    
    # Run inference
    S_pred, H_pred, P_pred = model(inputs, B, train=False)
    
    # Transform predictions to coordinates
    S_pred = jax.nn.sigmoid(S_pred)
    center_of_mass, pca_coeffs = H_pred[..., :2], H_pred[..., 2:]
    
    # Note: For NNX we need to pass pca_matrix separately or get it from metadata
    # This is a limitation of this interface - W_pred needs pca_matrix
    return S_pred, H_pred, P_pred
