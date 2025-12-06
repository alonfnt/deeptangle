"""Detect worms in video files - supports both Haiku and NNX models."""
from pathlib import Path

from absl import app, flags
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist

# scikit-video uses deprecated numpy.float, numpy.int
# hacky fix: https://github.com/scikit-video/scikit-video/issues/154
np.float = np.float64
np.int = np.int_
import skvideo.io

from deeptangle.inference import load_model, detect_model_format
from deeptangle.dataset.pca import points_from_pca
from deeptangle import time_activity


flags.DEFINE_string("input", default=None, required=True, help="Path to the video.")
flags.DEFINE_string("output", default="out.png", help="File where the output is saved.")
flags.DEFINE_string("model", default=None, required=True, help="Path to the model checkpoint directory.")
flags.DEFINE_float("correction_factor", default=1, help="Value of the correction_factor.")
flags.DEFINE_float("score_threshold", default=0.5, help="Score threshold to prune bad predictions.")
flags.DEFINE_integer("frame", default=5, help="Target frame to detect")
FLAGS = flags.FLAGS


def preprocess_clip(clip, correction_factor=1.0):
    """Preprocess video clip for inference."""
    # Invert colors
    clip = 255 - clip
    # Adaptive histogram equalization
    clip = equalize_adapthist(clip)
    # Apply correction factor
    clip = correction_factor * clip
    return clip


def detect_with_haiku(forward_fn, state, clip, score_threshold=0.5):
    """Run detection with Haiku model."""
    # Haiku models expect batch dimension
    if clip.ndim == 3:
        clip = clip[None, ...]
    
    # Run inference
    predictions, _ = forward_fn.apply(state[0], state[1], clip, False)
    
    # Filter by score threshold
    mask = predictions.s[0] > score_threshold
    w_filtered = predictions.w[0, mask]
    s_filtered = predictions.s[0, mask]
    
    return w_filtered, s_filtered


def detect_with_nnx(model, clip, pca_matrix, B, score_threshold=0.5):
    """Run detection with NNX model."""
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
    W_pred = points_from_pca(pca_coeffs, pca_matrix, center_of_mass)
    
    # Filter by score threshold
    mask = S_pred[0] > score_threshold
    w_filtered = W_pred[0, mask]
    s_filtered = S_pred[0, mask]


def main(args):
    del args
    
    with time_activity("Loading Model"):
        model_format = detect_model_format(FLAGS.model)
        print(f"Detected {model_format.upper()} model format")
        
        model_data = load_model(FLAGS.model)
        
        if model_format == "haiku":
            forward_fn, state, pca_matrix, metadata = model_data
        else:  # nnx
            model, pca_matrix, B, metadata = model_data
        
        print(f"Model expects {metadata['nframes']} frames")
    
    with time_activity("Reading input clip from video file"):
        frames_to_load = FLAGS.frame + metadata['nframes'] // 2 + 1
        initial_frame = max(0, FLAGS.frame - metadata['nframes'] // 2)
        video = skvideo.io.vread(FLAGS.input, num_frames=frames_to_load, as_grey=True)
        clip = video[initial_frame:initial_frame + metadata['nframes'], ..., 0]
    
    with time_activity("Pre-processing the clip"):
        clip = preprocess_clip(clip, FLAGS.correction_factor)
    
    with time_activity("Detecting worms"):
        if model_format == "haiku":
            worms, scores = detect_with_haiku(forward_fn, state, clip, FLAGS.score_threshold)
        else:  # nnx
            worms, scores = detect_with_nnx(model, clip, pca_matrix, B, FLAGS.score_threshold)
        print(f"Detected {len(worms)} worms")
    
    with time_activity("Plotting results"):
        plt.style.use("fast")
        plt.figure(figsize=(10, 10))
        plt.xlim(0, clip.shape[1])
        plt.ylim(0, clip.shape[0])
        
        # Show middle frame
        mid_frame = metadata['nframes'] // 2
        plt.imshow(clip[mid_frame], cmap="binary")
        
        # Plot detected worms
        for worm in worms:
            # Plot middle temporal frame
            skeleton = worm[1]  # temporal_window=3, so index 1 is middle
            plt.plot(skeleton[:, 0], skeleton[:, 1], "-", linewidth=2, alpha=0.8)
        
        plt.title(f"Detected {len(worms)} worms (frame {FLAGS.frame}, {model_format.upper()} model)")
        plt.savefig(FLAGS.output, dpi=300, bbox_inches='tight')
        print(f"Saved output to {FLAGS.output}")

if __name__ == "__main__":
    app.run(main)
