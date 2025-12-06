"""
Script to generate synthetic training datasets.

This script separates dataset generation from model training, allowing
the synthetic data to be pre-generated and saved to disk. This is the
first step towards making the model compatible with HuggingFace Hub.
"""
from pathlib import Path
from functools import partial

from absl import app, flags
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm

from celegans import sim_pca, simulate, video_synthesis
from celegans.transforms import random_gamma
from deeptangle.dataset import pca, transforms
import dm_pix as pix

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_samples", 10000, "Total number of training samples to generate.")
flags.DEFINE_integer("num_val_samples", 1000, "Number of validation samples to generate.")
flags.DEFINE_integer("nframes", 11, "Number of frames in a clip.")
flags.DEFINE_integer("size", 256, "Size of the frame for training.")
flags.DEFINE_list("nworms", [5, 10, 50, 100, 150, 200, 250], "Number of worms.")
flags.DEFINE_float("clip_duration", 0.55, "Seconds of simulation that the clip should last.")
flags.DEFINE_integer("kpoints", 49, "Number of points in the skeleton simulation.")
flags.DEFINE_integer("npca", 12, "Number of components of the pca.")
flags.DEFINE_integer("nworms_pca", 100_000, "Number of worms used to find the pca.")
flags.DEFINE_string("output_dir", "synthetic_dataset", "Directory to save the generated dataset.")
flags.DEFINE_bool("augmentation", True, "Apply augmentation to the training data.")


def normalization(image):
    """Normalize image to [0, 1] range based on percentiles."""
    lower = jnp.percentile(image, 1)
    upper = jnp.percentile(image, 99)
    return (image - lower) / (upper - lower)


def get_augmentations():
    """Define the augmentations to apply after the frames have been synthesized."""
    augments = [
        lambda key, image: pix.random_brightness(key, image, max_delta=0.2),
        lambda key, image: pix.random_contrast(key, image, lower=0.2, upper=0.5),
        lambda key, image: random_gamma(
            key, image, lower=0.5, upper=2.5, loc=-0.2, scale=1.0
        ),
    ]
    
    def augmentation_wrapper(f):
        def wrapper(key, frame):
            key, noise_key = jax.random.split(key)
            frame = transforms.add_channel(frame)
            frame = transforms.apply_random_white_noise(noise_key, frame, mu=0, p=0.2, maxstd=0.1)
            frame = jnp.clip(frame, 0.0, None)
            frame = f(key, frame)
            frame = transforms.remove_channel(frame)
            frame = normalization(frame)
            return frame
        return wrapper
    
    augments = list(map(augmentation_wrapper, augments))
    augments_p = jnp.array([0.2, 0.2, 0.6])
    return augments, augments_p


def generate_sample(key, nworms, augments=None, augments_p=None, apply_aug=True):
    """Generate a single synthetic sample (clip + label)."""
    sim_key, video_key, aug_key = jr.split(key, 3)
    
    # Simulate worm coordinates
    W = simulate(
        sim_key, nworms, FLAGS.clip_duration, FLAGS.nframes, FLAGS.size, FLAGS.kpoints
    )
    
    # Extract temporal window for label
    temporal_window = slice(len(W) // 2 - 1, len(W) // 2 + 2, 1)
    label = W[temporal_window, ...].transpose((1, 0, 2, 3))
    
    # Generate video frames
    X = video_synthesis(video_key, W, size=FLAGS.size)
    
    # Apply augmentation if specified
    if apply_aug and augments is not None:
        idx = jr.choice(aug_key, len(augments), p=augments_p)
        X = augments[int(idx)](aug_key, X)
    
    return X, label


def generate_pca_matrix(key):
    """Generate PCA transformation matrix."""
    print("Generating PCA matrix...")
    
    # Create a simple generator for PCA initialization
    class PCAGenerator:
        def __init__(self):
            pass
        
        def init_pca(self, key):
            return sim_pca(key, nworms=FLAGS.nworms_pca, kpoints=FLAGS.kpoints)
    
    pca_gen = PCAGenerator()
    A = pca.init_pca(key, pca_gen, n_components=FLAGS.npca)
    print(f"PCA matrix shape: {A.shape}")
    return A


def save_dataset(output_dir, X_train, y_train, X_val, y_val, A, metadata):
    """Save generated dataset to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to {output_dir}...")
    
    # Save training data as individual files (due to variable shapes)
    train_dir = output_path / "train"
    train_dir.mkdir(exist_ok=True)
    for i, (X, y) in enumerate(zip(X_train, y_train)):
        np.savez_compressed(
            train_dir / f"sample_{i:06d}.npz",
            X=np.array(X),
            y=np.array(y)
        )
    
    # Save validation data as individual files
    val_dir = output_path / "val"
    val_dir.mkdir(exist_ok=True)
    for i, (X, y) in enumerate(zip(X_val, y_val)):
        np.savez_compressed(
            val_dir / f"sample_{i:06d}.npz",
            X=np.array(X),
            y=np.array(y)
        )
    
    # Save PCA matrix
    np.save(output_path / "pca_matrix.npy", np.array(A))
    
    # Save metadata
    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Dataset saved successfully!")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Clip shape: {X_train[0].shape}")
    print(f"  - Label shape: {y_train[0].shape}")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    print("=" * 60)
    print("Synthetic Dataset Generation")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Random seed: {FLAGS.seed}")
    print(f"  - Training samples: {FLAGS.num_samples}")
    print(f"  - Validation samples: {FLAGS.num_val_samples}")
    print(f"  - Frame size: {FLAGS.size}")
    print(f"  - Frames per clip: {FLAGS.nframes}")
    print(f"  - Worm counts: {FLAGS.nworms}")
    print(f"  - Output directory: {FLAGS.output_dir}")
    print("=" * 60)
    
    # Initialize random keys
    key = jr.PRNGKey(FLAGS.seed)
    pca_key, train_key, val_key = jr.split(key, 3)
    
    # Generate PCA matrix
    A = generate_pca_matrix(pca_key)
    
    # Prepare augmentations
    augments, augments_p = get_augmentations() if FLAGS.augmentation else (None, None)
    
    # Generate training data
    print("\nGenerating training data...")
    X_train, y_train = [], []
    nworms_list = list(map(int, FLAGS.nworms))
    
    for i in tqdm(range(FLAGS.num_samples)):
        sample_key = jr.fold_in(train_key, i)
        nworms_key, gen_key = jr.split(sample_key)
        
        # Randomly select number of worms
        nworms = nworms_list[int(jr.randint(nworms_key, (), 0, len(nworms_list)))]
        
        # Generate sample
        X, y = generate_sample(gen_key, nworms, augments, augments_p, apply_aug=True)
        X_train.append(X)
        y_train.append(y)
    
    # Generate validation data
    print("\nGenerating validation data...")
    X_val, y_val = [], []
    
    for i in tqdm(range(FLAGS.num_val_samples)):
        sample_key = jr.fold_in(val_key, i)
        nworms_key, gen_key = jr.split(sample_key)
        
        # Randomly select number of worms
        nworms = nworms_list[int(jr.randint(nworms_key, (), 0, len(nworms_list)))]
        
        # Generate sample (no augmentation for validation)
        X, y = generate_sample(gen_key, nworms, augments, augments_p, apply_aug=False)
        X_val.append(X)
        y_val.append(y)
    
    # Prepare metadata
    metadata = {
        "seed": FLAGS.seed,
        "num_train_samples": FLAGS.num_samples,
        "num_val_samples": FLAGS.num_val_samples,
        "nframes": FLAGS.nframes,
        "size": FLAGS.size,
        "nworms": nworms_list,
        "clip_duration": FLAGS.clip_duration,
        "kpoints": FLAGS.kpoints,
        "npca": FLAGS.npca,
        "nworms_pca": FLAGS.nworms_pca,
        "augmentation": FLAGS.augmentation,
    }
    
    # Save dataset
    save_dataset(FLAGS.output_dir, X_train, y_train, X_val, y_val, A, metadata)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    app.run(main)
