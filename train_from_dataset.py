"""
Training script that uses pre-generated synthetic datasets.

This script separates training from dataset generation, making the pipeline
more modular and HuggingFace-compatible.
"""
from collections import namedtuple
from functools import partial
from pathlib import Path

from absl import app, flags
import flax
from flax.training import train_state
import jax
from jax import jit, vmap, grad
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
import jax.random as jr
import optax

from deeptangle.dataset.loader import SyntheticDatasetLoader, load_pca_matrix, load_metadata
from deeptangle.dataset.pca import points_from_pca
from deeptangle.model_flax import create_detector
from deeptangle import logger, checkpoints_flax

Losses = namedtuple("Losses", ["w", "s", "p"])

FLAGS = flags.FLAGS
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
flags.DEFINE_string("checkpoint_dir", "checkpoints_flax", "Path to the dir containing checkpoints.")
flags.DEFINE_bool("save", False, "Whether to save checkpoints.")
flags.DEFINE_integer("save_interval", 1000, "Steps between checkpoint saves.")
flags.DEFINE_integer("cutoff", 48, "Distance cutoff for latent space visibility.")


class TrainState(train_state.TrainState):
    """Extended training state with batch statistics."""
    batch_stats: flax.core.FrozenDict


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


def create_train_state(rng, model, learning_rate, input_shape, B):
    """Create initial training state."""
    # Initialize model
    # input_shape is (batch, time, height, width)
    # We need (batch, height, width, time) for the model
    batch, time, height, width = input_shape
    dummy_input = jnp.ones((1, height, width, time))  # (1, H, W, C)
    variables = model.init(rng, dummy_input, B, train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    
    # Create optimizer
    tx = optax.adamw(learning_rate=learning_rate)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )


@jit
def train_step(state, batch_X, batch_y, B, A, size):
    """Perform a single training step."""
    
    def loss_fn(params):
        # Transpose batch for model input (B, T, H, W) -> (B, H, W, T)
        inputs = jnp.transpose(batch_X, axes=(0, 2, 3, 1))
        
        # Forward pass
        variables = {'params': params, 'batch_stats': state.batch_stats}
        outputs, new_model_state = state.apply_fn(
            variables, inputs, B, train=True, mutable=['batch_stats']
        )
        
        S_pred, H_pred, P_pred = outputs
        
        # Transform predictions to coordinates
        CM, H = H_pred[..., :2], H_pred[..., 2:]
        W_pred = points_from_pca(H, A, CM)
        
        # Apply sigmoid to scores
        S_pred = jax.nn.sigmoid(S_pred)
        
        predictions = (W_pred, S_pred, P_pred)
        
        # Compute losses (average over batch since batch_y is a list)
        total_loss = 0.0
        count = 0
        for i in range(len(batch_y)):
            if batch_y[i] is not None:
                losses = multi_loss_fn(
                    tuple(p[i:i+1] for p in predictions),
                    batch_y[i][None, ...],
                    size,
                    FLAGS.sigma,
                    FLAGS.cutoff
                )
                weights = Losses(FLAGS.wloss_w, FLAGS.wloss_s, FLAGS.wloss_p)
                weighted_losses = tree_map(jnp.multiply, weights, losses)
                loss = tree_reduce(jnp.add, weighted_losses)
                total_loss += loss
                count += 1
        
        total_loss = total_loss / max(count, 1)
        return total_loss, (new_model_state, losses)
    
    (loss, (new_model_state, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state.get('batch_stats', state.batch_stats)
    )
    
    return state, loss, losses


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    print("=" * 60)
    print("Training with Pre-Generated Dataset")
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
    print("\nInitializing model...")
    kpoints2 = A.shape[1]
    J = jnp.flip(jnp.identity(kpoints2), axis=1)
    B = A @ J @ jnp.transpose(A)
    
    model = create_detector(
        npoints=npca,
        n_suggestions=FLAGS.n_suggestions,
        latent_dim=FLAGS.latent_dim,
        nframes=nframes
    )
    
    # Initialize training state
    rng = jr.PRNGKey(FLAGS.seed)
    init_rng, train_rng = jr.split(rng)
    
    input_shape = (FLAGS.batch_size, nframes, size, size)
    state = create_train_state(init_rng, model, FLAGS.learning_rate, input_shape, B)
    
    print("Model initialized successfully!")
    print("=" * 60)
    
    # Create checkpoint directory
    checkpoint_dir = Path(FLAGS.checkpoint_dir).absolute()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    train_iter = train_loader.get_batch_iterator(FLAGS.batch_size, shuffle=True, seed=FLAGS.seed)
    
    best_loss = float('inf')
    
    for step in range(FLAGS.train_steps):
        # Get batch
        batch_X, batch_y = next(train_iter)
        
        # Training step
        state, loss, losses = train_step(state, batch_X, batch_y, B, A, size)
        
        # Logging
        if (step + 1) % FLAGS.eval_interval == 0:
            loss_val = float(loss)
            print(f"Step {step + 1}/{FLAGS.train_steps}: "
                  f"Loss = {loss_val:.4f}, "
                  f"w = {float(losses.w):.4f}, "
                  f"s = {float(losses.s):.4f}, "
                  f"p = {float(losses.p):.4f}")
            
            # Save if best
            if FLAGS.save and loss_val < best_loss:
                best_loss = loss_val
                print(f"  -> Saving best checkpoint (loss: {best_loss:.4f})")
                checkpoints_flax.save_checkpoint(
                    checkpoint_dir=str(checkpoint_dir),
                    state=state,
                    step=step + 1,
                    keep=3,
                    overwrite=False
                )
                # Also save in inference format
                model_metadata = {
                    "nframes": nframes,
                    "size": size,
                    "npca": npca,
                    "n_suggestions": FLAGS.n_suggestions,
                    "latent_dim": FLAGS.latent_dim,
                    "step": step + 1,
                    "loss": loss_val,
                }
                checkpoints_flax.save_model_for_inference(
                    output_dir=str(checkpoint_dir / "best_model"),
                    params=state.params,
                    batch_stats=state.batch_stats,
                    pca_matrix=A,
                    metadata=model_metadata
                )
        
        # Periodic save
        if FLAGS.save and (step + 1) % FLAGS.save_interval == 0:
            print(f"  -> Saving periodic checkpoint at step {step + 1}")
            checkpoints_flax.save_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                state=state,
                step=step + 1,
                keep=3,
                overwrite=False
            )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    app.run(main)
