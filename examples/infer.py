"""Detect worms in image sequences."""
from pathlib import Path
import json

from absl import app, flags
from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as skio
from skimage.exposure import equalize_adapthist

from deeptangle.model import create_detector
from deeptangle.dataset.pca import points_from_pca


flags.DEFINE_string("images", default=None, required=True, 
                    help="Directory containing image sequence or pattern (e.g., 'frames/*.png')")
flags.DEFINE_string("model", default=None, required=True, 
                    help="Path to the model checkpoint directory.")
flags.DEFINE_string("output", default="inference_result.png", 
                    help="Output image file.")
flags.DEFINE_float("score_threshold", default=0.5, 
                   help="Score threshold for filtering detections.")
flags.DEFINE_integer("start_frame", default=0, 
                    help="Starting frame index.")
FLAGS = flags.FLAGS


def load_image_sequence(pattern: str, start_frame: int, num_frames: int):
    """Load a sequence of images."""
    import glob
    
    # Get all matching files
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    # Load frames
    frames = []
    for i in range(start_frame, min(start_frame + num_frames, len(files))):
        img = skio.imread(files[i], as_gray=True)
        frames.append(img)
    
    if len(frames) < num_frames:
        raise ValueError(f"Not enough frames. Need {num_frames}, found {len(frames)}")
    
    return np.stack(frames)


def load_model(model_dir: str):
    """Load trained NNX model."""
    model_path = Path(model_dir)
    
    # Load metadata
    with open(model_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load PCA matrix
    pca_matrix = jnp.array(np.load(model_path / "pca_matrix.npy"))
    
    # Create model
    rngs = nnx.Rngs(0)
    model = create_detector(
        npoints=metadata["npca"],
        n_suggestions=metadata["n_suggestions"],
        latent_dim=metadata["latent_dim"],
        nframes=metadata["nframes"],
        rngs=rngs
    )
    
    # Load weights
    with open(model_path / "model.nnx", "rb") as f:
        model = nnx.from_bytes(model, f.read())
    
    # Compute B matrix
    kpoints2 = pca_matrix.shape[1]
    J = jnp.flip(jnp.identity(kpoints2), axis=1)
    B = pca_matrix @ J @ jnp.transpose(pca_matrix)
    
    return model, pca_matrix, B, metadata


def preprocess(frames):
    """Preprocess frames for inference."""
    # Normalize to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    
    # Adaptive histogram equalization
    processed = []
    for frame in frames:
        processed.append(equalize_adapthist(frame))
    
    return np.stack(processed)


def run_inference(model, frames, pca_matrix, B, score_threshold):
    """Run model inference."""
    # Add batch dimension and transpose to (B, H, W, T)
    inputs = jnp.transpose(frames[None, ...], axes=(0, 2, 3, 1))
    
    # Inference
    S_pred, H_pred, P_pred = model(inputs, B, train=False)
    
    # Post-process predictions
    S_pred = jax.nn.sigmoid(S_pred)
    center_of_mass, pca_coeffs = H_pred[..., :2], H_pred[..., 2:]
    W_pred = points_from_pca(pca_coeffs, pca_matrix, center_of_mass)
    
    # Filter by score
    mask = S_pred[0] > score_threshold
    detections = {
        'worms': W_pred[0, mask],
        'scores': S_pred[0, mask],
        'centers': center_of_mass[0, mask],
    }
    
    return detections


def visualize_results(frames, detections, output_path):
    """Visualize detection results."""
    # Use middle frame for visualization
    mid_idx = len(frames) // 2
    frame = frames[mid_idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(frame, cmap='gray')
    
    # Plot each detected worm
    colors = plt.cm.rainbow(np.linspace(0, 1, len(detections['worms'])))
    
    for i, (worm, score) in enumerate(zip(detections['worms'], detections['scores'])):
        # Plot skeleton from middle temporal frame
        skeleton = worm[1]  # temporal_window=3, middle frame
        ax.plot(skeleton[:, 0], skeleton[:, 1], '-', 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'Worm {i+1} (score: {score:.2f})')
    
    ax.set_title(f'Detected {len(detections["worms"])} worms')
    ax.axis('off')
    
    if len(detections['worms']) < 20:  # Only show legend if not too many detections
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")


def main(args):
    del args
    
    print("=" * 60)
    print("DeepTangle NNX Inference")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {FLAGS.model}...")
    model, pca_matrix, B, metadata = load_model(FLAGS.model)
    print(f"Model loaded successfully")
    print(f"  - Expects {metadata['nframes']} frames")
    print(f"  - Image size: {metadata['size']}x{metadata['size']}")
    
    # Load images
    print(f"\nLoading images from {FLAGS.images}...")
    frames = load_image_sequence(
        FLAGS.images, 
        FLAGS.start_frame, 
        metadata['nframes']
    )
    print(f"Loaded {len(frames)} frames")
    print(f"  - Frame shape: {frames[0].shape}")
    
    # Preprocess
    print("\nPreprocessing...")
    frames = preprocess(frames)
    
    # Run inference
    print("\nRunning inference...")
    detections = run_inference(model, frames, pca_matrix, B, FLAGS.score_threshold)
    print(f"Detected {len(detections['worms'])} worms")
    print(f"  - Score threshold: {FLAGS.score_threshold}")
    print(f"  - Mean score: {float(detections['scores'].mean()):.3f}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_results(frames, detections, FLAGS.output)
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    app.run(main)
