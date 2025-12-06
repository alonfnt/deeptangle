"""
Training script with Flax NNX models, config file support, and TensorBoard logging.

This script uses pre-generated synthetic datasets and the new Flax NNX API.
"""
from collections import namedtuple
from pathlib import Path
import yaml

from absl import app, flags
from flax import nnx
import jax
from jax import jit, vmap, grad
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
import jax.random as jr
import optax
from tensorboardX import SummaryWriter

from deeptangle.dataset.loader import SyntheticDatasetLoader, load_pca_matrix, load_metadata
from deeptangle.dataset.pca import points_from_pca
from deeptangle.model_nnx import create_detector
from deeptangle import logger

Losses = namedtuple("Losses", ["w", "s", "p"])

FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Path to YAML config file. If provided, overrides command-line args.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 8, "Size of the training batch.")
flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
flags.DEFINE_integer("train_steps", 100_000, "Number of training steps.")
flags.DEFINE_integer("eval_interval", 100, "Number of steps between evaluations.")
flags.DEFINE_string("dataset_dir", "synthetic_dataset", "Directory containing the pre-generated dataset.")
flags.DEFINE_integer("n_suggestions", 8, "Number of suggestions for cell in last layer.")
flags.DEFINE_integer("latent_dim", 8, "Dimension of the latent space.")
flags.DEFINE_float("sigma", 10, "Smoothness of the score")
flags.DEFINE_float("wloss_w", 1, "Weight on the coordinates loss.")
flags.DEFINE_float("wloss_s", 1e2, "Weight on the score/confidence loss.")
flags.DEFINE_float("wloss_p", 1e5, "Weight on the latent space loss.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Path to the dir containing checkpoints.")
flags.DEFINE_bool("save", False, "Whether to save checkpoints.")
flags.DEFINE_integer("save_interval", 1000, "Steps between checkpoint saves.")
flags.DEFINE_integer("cutoff", 48, "Distance cutoff for latent space visibility.")
flags.DEFINE_string("tensorboard_dir", "runs", "Directory for TensorBoard logs.")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_flags(config: dict):
    """Merge config file values with FLAGS, config takes precedence."""
    for key, value in config.items():
        if hasattr(FLAGS, key):
            FLAGS[key].value = value


def _importance_weights(n: int) -> jnp.ndarray:
    """Compute importance weights for temporal frames."""
    weights = 1 / (jnp.abs(jnp.arange(-n // 2 + 1, n // 2 + 1)) + 1)
    return weights / weights.sum()


def multi_loss_fn(Y_pred, Y_label, size, sigma, cutoff):
    """Compute multi-task loss."""
    X_pred, S_pred, P_pred = Y_pred
    
    inside = jnp.all((Y_label >= 0) & (Y_label < size), axis=(-1, -2, -3))
    
    @vmap
    def distance_matrix(a, b):
        A = a[None, ...]
        B = b[:, None, ...]
        return jnp.sum((A - B) ** 2, axis=-1)
    
    # Compute distance matrix for direct and flip versions
    distance = distance_matrix(X_pred, Y_label).mean(-1)
    flip_distance = distance_matrix(X_pred, jnp.flip(Y_label, axis=-2)).mean(-1)
    distances = jnp.minimum(distance, flip_distance)
    
    # Weight by importance of each frame
    num_frames = X_pred.shape[2]
    temporal_weights = _importance_weights(num_frames)
    distances = jnp.average(distances, axis=-1, weights=temporal_weights)
    
    # Compute loss for points inside the frame
    inside_count = jnp.sum(inside) + 1e-6
    masked_distances = distances * inside[:, :, None]
    Loss_X = jnp.sum(jnp.min(masked_distances, axis=2)) / inside_count
    
    # Stop gradients for score and latent space losses
    distances = jax.lax.stop_gradient(distances)
    X = jax.lax.stop_gradient(X_pred)
    
    # Compute confidence scores
    scores = jnp.exp(-jnp.min(distances, axis=1) / sigma)
    Loss_S = jnp.mean((scores - S_pred) ** 2)
    
    # Find closest target for each prediction
    T = jnp.argmin(distances, axis=1)
    same_T = T[:, None, :] == T[:, :, None]
    
    # Visibility mask for latent space
    distance_ls = distance_matrix(P_pred, P_pred)
    K = X.shape[3]
    Xcm = X[:, :, num_frames // 2, K // 2, :]
    visible = distance_matrix(Xcm, Xcm) < cutoff**2
    factor = visible / (visible.sum(axis=2)[:, :, None] + 1e-6)
    
    # Compute latent space loss
    safe_log = lambda x: jnp.log(jnp.where(x > 0.0, x, 1.0))
    attraction = distance_ls
    repulsion = -safe_log(1 - jnp.exp(-distance_ls))
    Loss_P = factor * jnp.where(same_T, attraction, repulsion)
    
    # Weight by scores
    scores_matrix = scores[:, :, None] * scores[:, None, :]
    Loss_P = jnp.sum(scores_matrix * Loss_P) / (scores_matrix.sum() + 1e-6)
    
    return Losses(Loss_X, Loss_S, Loss_P)


@jit
def train_step(model, optimizer, batch_X, batch_y, B, A, size, wloss_w, wloss_s, wloss_p, sigma, cutoff):
    """Perform a single training step."""
    
    def loss_fn(model):
        # Transpose batch for model input (B, T, H, W) -> (B, H, W, T)
        inputs = jnp.transpose(batch_X, axes=(0, 2, 3, 1))
        
        # Forward pass
        S_pred, H_pred, P_pred = model(inputs, B, train=True)
        
        # Transform predictions to coordinates
        center_of_mass, pca_coeffs = H_pred[..., :2], H_pred[..., 2:]
        W_pred = points_from_pca(pca_coeffs, A, center_of_mass)
        
        # Apply sigmoid to scores
        S_pred = jax.nn.sigmoid(S_pred)
        
        predictions = (W_pred, S_pred, P_pred)
        
        # Compute losses (average over batch since batch_y is a list)
        total_loss = 0.0
        count = 0
        losses_accum = Losses(0.0, 0.0, 0.0)
        
        for i in range(len(batch_y)):
            if batch_y[i] is not None:
                losses = multi_loss_fn(
                    tuple(p[i:i+1] for p in predictions),
                    batch_y[i][None, ...],
                    size,
                    sigma,
                    cutoff
                )
                weights = Losses(wloss_w, wloss_s, wloss_p)
                weighted_losses = tree_map(jnp.multiply, weights, losses)
                loss = tree_reduce(jnp.add, weighted_losses)
                total_loss += loss
                losses_accum = tree_map(jnp.add, losses_accum, losses)
                count += 1
        
        total_loss = total_loss / max(count, 1)
        avg_losses = tree_map(lambda x: x / max(count, 1), losses_accum)
        return total_loss, avg_losses
    
    loss, grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    total_loss, losses = loss
    
    # Update parameters
    optimizer.update(model, grads)
    
    return total_loss, losses


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    # Load config file if provided
    if FLAGS.config:
        print(f"Loading configuration from {FLAGS.config}")
        config = load_config(FLAGS.config)
        merge_config_with_flags(config)
    
    print("=" * 60)
    print("Training with Pre-Generated Dataset (Flax NNX)")
    print("=" * 60)
    
    # Load dataset and metadata
    print(f"Loading dataset from {FLAGS.dataset_dir}...")
    train_loader = SyntheticDatasetLoader(FLAGS.dataset_dir, split="train")
    val_loader = SyntheticDatasetLoader(FLAGS.dataset_dir, split="val")
    
    metadata = load_metadata(FLAGS.dataset_dir)
    A = load_pca_matrix(FLAGS.dataset_dir)
    
    # Extract configuration from metadata
    nframes = metadata["nframes"]
    size = metadata["size"]
    npca = metadata["npca"]
    
    print(f"Dataset configuration:")
    print(f"  - Frames: {nframes}")
    print(f"  - Size: {size}")
    print(f"  - PCA components: {npca}")
    print(f"  - Training samples: {len(train_loader)}")
    print(f"  - Validation samples: {len(val_loader)}")
    
    # Create model
    print("\nInitializing Flax NNX model...")
    kpoints2 = A.shape[1]
    J = jnp.flip(jnp.identity(kpoints2), axis=1)
    B = A @ J @ jnp.transpose(A)
    
    # Initialize model with NNX
    rngs = nnx.Rngs(FLAGS.seed)
    model = create_detector(
        npoints=npca,
        n_suggestions=FLAGS.n_suggestions,
        latent_dim=FLAGS.latent_dim,
        nframes=nframes,
        rngs=rngs
    )
    
    # Initialize optimizer
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=FLAGS.learning_rate), wrt=nnx.Param)
    
    print("Model initialized successfully!")
    print("=" * 60)
    
    # Create checkpoint and tensorboard directories
    checkpoint_dir = Path(FLAGS.checkpoint_dir).absolute()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_dir = Path(FLAGS.tensorboard_dir).absolute()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training loop
    print("\nStarting training...")
    train_iter = train_loader.get_batch_iterator(FLAGS.batch_size, shuffle=True, seed=FLAGS.seed)
    
    best_loss = float('inf')
    
    for step in range(FLAGS.train_steps):
        # Get batch
        batch_X, batch_y = next(train_iter)
        
        # Training step
        loss, losses = train_step(
            model, optimizer, batch_X, batch_y, B, A, size,
            FLAGS.wloss_w, FLAGS.wloss_s, FLAGS.wloss_p,
            FLAGS.sigma, FLAGS.cutoff
        )
        
        # Logging
        if (step + 1) % FLAGS.eval_interval == 0:
            loss_val = float(loss)
            w_loss = float(losses.w)
            s_loss = float(losses.s)
            p_loss = float(losses.p)
            
            print(f"Step {step + 1}/{FLAGS.train_steps}: "
                  f"Loss = {loss_val:.4f}, "
                  f"w = {w_loss:.4f}, "
                  f"s = {s_loss:.4f}, "
                  f"p = {p_loss:.4f}")
            
            # Log to TensorBoard
            writer.add_scalar('loss/total', loss_val, step + 1)
            writer.add_scalar('loss/coordinates', w_loss, step + 1)
            writer.add_scalar('loss/score', s_loss, step + 1)
            writer.add_scalar('loss/latent', p_loss, step + 1)
            
            # Save if best
            if FLAGS.save and loss_val < best_loss:
                best_loss = loss_val
                print(f"  -> Saving best checkpoint (loss: {best_loss:.4f})")
                
                # Save NNX model
                checkpoint_path = checkpoint_dir / f"best_model_step_{step + 1}"
                checkpoint_path.mkdir(exist_ok=True)
                
                # Save model state
                with open(checkpoint_path / "model.nnx", "wb") as f:
                    f.write(nnx.to_bytes(model))
                
                # Save metadata
                import json
                model_metadata = {
                    "nframes": nframes,
                    "size": size,
                    "npca": npca,
                    "n_suggestions": FLAGS.n_suggestions,
                    "latent_dim": FLAGS.latent_dim,
                    "step": step + 1,
                    "loss": loss_val,
                }
                with open(checkpoint_path / "metadata.json", "w") as f:
                    json.dump(model_metadata, f, indent=2)
                
                # Save PCA matrix
                import numpy as np
                np.save(checkpoint_path / "pca_matrix.npy", np.array(A))
        
        # Periodic save
        if FLAGS.save and (step + 1) % FLAGS.save_interval == 0:
            print(f"  -> Saving periodic checkpoint at step {step + 1}")
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step + 1}"
            checkpoint_path.mkdir(exist_ok=True)
            
            with open(checkpoint_path / "model.nnx", "wb") as f:
                f.write(nnx.to_bytes(model))
    
    # Close TensorBoard writer
    writer.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    app.run(main)
