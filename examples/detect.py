"""Detect worms in video files."""
from pathlib import Path
import json

from absl import app, flags
from flax import nnx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist

# scikit-video uses deprecated numpy.float, numpy.int
# hacky fix: https://github.com/scikit-video/scikit-video/issues/154
np.float = np.float64
np.int = np.int_
import skvideo.io

from deeptangle.model import create_detector
from deeptangle.dataset.pca import points_from_pca
from deeptangle import time_activity


flags.DEFINE_string("input", default=None, required=True, help="Path to the video.")
flags.DEFINE_string("output", default="out.png", help="File where the output is saved.")
flags.DEFINE_string("model", default=None, required=True, help="Path to the model checkpoint directory.")
flags.DEFINE_float("correction_factor", default=1, help="Value of the correction_factor.")
flags.DEFINE_float("score_threshold", default=0.5, help="Score threshold to prune bad predictions.")
flags.DEFINE_integer("frame", default=5, help="Target frame to detect")
FLAGS = flags.FLAGS


def load_nnx_model(model_dir: str):
    """Load a trained NNX model from checkpoint."""
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
    
    return model, pca_matrix, B, metadata


def preprocess_clip(clip, correction_factor=1.0):
    """Preprocess video clip for inference."""
    # Invert colors
    clip = 255 - clip
    # Adaptive histogram equalization
    clip = equalize_adapthist(clip)
    # Apply correction factor
    clip = correction_factor * clip
    return clip


def detect_worms(model, clip, pca_matrix, B, score_threshold=0.5):
    """Run detection on a clip."""
    # Transpose to (B, H, W, T) format
    inputs = jnp.transpose(clip, axes=(0, 2, 3, 1))
    
    # Run inference
    S_pred, H_pred, P_pred = model(inputs, B, train=False)
    
    # Transform predictions to coordinates
    S_pred = jax.nn.sigmoid(S_pred)
    center_of_mass, pca_coeffs = H_pred[..., :2], H_pred[..., 2:]
    W_pred = points_from_pca(pca_coeffs, pca_matrix, center_of_mass)
    
    # Filter by score threshold
    mask = S_pred[0] > score_threshold
    w_filtered = W_pred[0, mask]
    s_filtered = S_pred[0, mask]
    
    return w_filtered, s_filtered


def main(args):
    del args
    
    with time_activity("Loading NNX Model"):
        model, pca_matrix, B, metadata = load_nnx_model(FLAGS.model)
        print(f"Loaded model from {FLAGS.model}")
        print(f"Model expects {metadata['nframes']} frames")
    
    with time_activity("Reading input clip from video file"):
        frames_to_load = FLAGS.frame + metadata['nframes'] // 2 + 1
        initial_frame = max(0, FLAGS.frame - metadata['nframes'] // 2)
        video = skvideo.io.vread(FLAGS.input, num_frames=frames_to_load, as_grey=True)
        clip = video[initial_frame:initial_frame + metadata['nframes'], ..., 0]
    
    with time_activity("Pre-processing the clip"):
        clip = preprocess_clip(clip, FLAGS.correction_factor)
        clip = clip[None, ...]  # Add batch dimension
    
    with time_activity("Detecting worms"):
        worms, scores = detect_worms(model, clip, pca_matrix, B, FLAGS.score_threshold)
        print(f"Detected {len(worms)} worms")
    
    with time_activity("Plotting results"):
        plt.style.use("fast")
        plt.figure(figsize=(10, 10))
        plt.xlim(0, clip.shape[3])
        plt.ylim(0, clip.shape[2])
        
        # Show middle frame
        mid_frame = metadata['nframes'] // 2
        plt.imshow(clip[0, mid_frame], cmap="binary")
        
        # Plot detected worms
        for worm in worms:
            # Plot middle temporal frame
            skeleton = worm[1]  # temporal_window=3, so index 1 is middle
            plt.plot(skeleton[:, 0], skeleton[:, 1], "-", linewidth=2, alpha=0.8)
        
        plt.title(f"Detected {len(worms)} worms (frame {FLAGS.frame})")
        plt.savefig(FLAGS.output, dpi=300, bbox_inches='tight')
        print(f"Saved output to {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
