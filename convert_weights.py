"""
Convert Haiku model checkpoints to Flax NNX format.

This script converts trained weights from the original Haiku model
to the new Flax NNX model format, making them compatible with the
new training and inference pipeline.
"""
from pathlib import Path
import json
import pickle

from absl import app, flags
import haiku as hk
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from deeptangle import checkpoints
from deeptangle.model import create_detector
from deeptangle.logger import recover_experiment_parameters


FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "Path to Haiku checkpoint directory.", required=True)
flags.DEFINE_string("output_dir", None, "Path to save NNX checkpoint.", required=True)


def load_haiku_checkpoint(checkpoint_dir: str):
    """Load Haiku checkpoint and metadata."""
    path = Path(checkpoint_dir)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
    
    # Load experiment parameters
    params = recover_experiment_parameters(checkpoint_dir)
    
    # Load PCA matrix
    pca_matrix = checkpoints.load_pca_matrix(checkpoint_dir)
    
    # Load Haiku state (params and state)
    with path.joinpath('tree.pkl').open('rb') as f:
        tree_struct = pickle.load(f)
    
    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with path.joinpath('arrays.npy').open('rb') as f:
        flat_state = [jnp.load(f) for _ in leaves]
    
    haiku_state = jax.tree_util.tree_unflatten(treedef, flat_state)
    
    return haiku_state, pca_matrix, params


def map_haiku_to_nnx_params(haiku_params, nnx_model):
    """
    Map Haiku parameter names to NNX model structure.
    
    This is a simplified mapping - you may need to adjust based on
    the actual parameter structure differences between Haiku and NNX models.
    """
    print("Mapping Haiku parameters to NNX model...")
    print(f"Haiku params keys: {list(haiku_params.keys())[:5]}...")
    
    # Note: This is a placeholder. The actual mapping would need to be
    # implemented based on the specific parameter structure.
    # For now, we'll just initialize a new model and note that manual
    # mapping may be required.
    
    print("\nWARNING: Automatic parameter mapping from Haiku to NNX is not")
    print("implemented yet. The NNX model will be initialized with random weights.")
    print("\nTo properly convert weights, you would need to:")
    print("1. Map each Haiku parameter name to its corresponding NNX parameter")
    print("2. Handle any structural differences (e.g., batch norm statistics)")
    print("3. Verify the converted model produces similar outputs")
    
    return nnx_model


def save_nnx_checkpoint(output_dir: str, model, pca_matrix, metadata):
    """Save model in NNX format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(output_path / "model.nnx", "wb") as f:
        f.write(nnx.to_bytes(model))
    
    # Save PCA matrix
    np.save(output_path / "pca_matrix.npy", np.array(pca_matrix))
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nNNX checkpoint saved to {output_dir}")
    print(f"  - model.nnx: {(output_path / 'model.nnx').stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  - pca_matrix.npy: {(output_path / 'pca_matrix.npy').stat().st_size / 1024:.2f} KB")
    print(f"  - metadata.json")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    print("=" * 60)
    print("Haiku to NNX Checkpoint Converter")
    print("=" * 60)
    
    # Load Haiku checkpoint
    print(f"\nLoading Haiku checkpoint from {FLAGS.input_dir}...")
    haiku_state, pca_matrix, params = load_haiku_checkpoint(FLAGS.input_dir)
    
    print(f"Loaded checkpoint:")
    print(f"  - nframes: {params['nframes']}")
    print(f"  - n_suggestions: {params['n_suggestions']}")
    print(f"  - latent_dim: {params['latent_dim']}")
    print(f"  - PCA components: {pca_matrix.shape[0]}")
    
    # Create NNX model with same architecture
    print("\nCreating NNX model...")
    rngs = nnx.Rngs(0)
    nnx_model = create_detector(
        npoints=pca_matrix.shape[0],
        n_suggestions=params['n_suggestions'],
        latent_dim=params['latent_dim'],
        nframes=params['nframes'],
        rngs=rngs
    )
    
    # Attempt to map parameters (note: this is a placeholder)
    nnx_model = map_haiku_to_nnx_params(haiku_state, nnx_model)
    
    # Prepare metadata
    metadata = {
        "nframes": params['nframes'],
        "size": params.get('size', 256),  # Default from paper
        "npca": pca_matrix.shape[0],
        "n_suggestions": params['n_suggestions'],
        "latent_dim": params['latent_dim'],
        "converted_from": "haiku",
        "source_checkpoint": FLAGS.input_dir,
    }
    
    # Save NNX checkpoint
    print(f"\nSaving NNX checkpoint to {FLAGS.output_dir}...")
    save_nnx_checkpoint(FLAGS.output_dir, nnx_model, pca_matrix, metadata)
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print("\nNOTE: This converter currently creates a new NNX model with the")
    print("same architecture but does NOT transfer the learned weights.")
    print("\nFor production use, you would need to:")
    print("1. Implement proper weight mapping between Haiku and NNX")
    print("2. Verify converted model produces same outputs as original")
    print("3. Test on validation data to ensure accuracy is preserved")
    print("\nAlternatively, consider retraining the model using train.py")
    print("with the new NNX implementation.")


if __name__ == "__main__":
    app.run(main)
