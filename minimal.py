"""
Simpler implementation of de(ep)tangle training with syntehtic data.

author: Albert Alonso (2024)

===
Some things are still missing to be implemented:
    - Add checkpoint manager / saving weights.
    - Add evaluation loop.
    - Add logging.
    - Add data augmentation and normalization.
    - Incorporate the PCA step into the model.
    - Add distributed training.
"""
from typing import Sequence

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm
import pcax

from celegans import sim_pca, simulate, video_synthesis


@struct.dataclass
class CElegansDataset:
    num_worms: int = 50
    num_points: int = 49
    num_frames: int = 11
    batch_size: int = 4
    box_size: int = 256
    video_size: int = 256
    duration: float = 0.55

    def get_batch(self, key):
        batch_keys = jax.random.split(key, self.batch_size)

        def generate_sample(key):
            sim_key, video_key = jax.random.split(key)
            labels = simulate(sim_key, self.num_worms, self.duration, self.num_frames, self.box_size, self.num_points)
            frames = video_synthesis(video_key, labels, size=self.video_size)
            labels = jnp.swapaxes(labels, 0, 1)
            labels = labels[:, self.num_frames // 2 - 1 : self.num_frames // 2 + 2]
            return frames, labels

        return jax.vmap(generate_sample)(batch_keys)

    def debug_sample(self, key):
        frames, _ = self.get_batch(key)
        plt.figure(figsize=(5, 5))
        plt.imshow(frames[0, 0], cmap="gray")
        plt.title("Sample Synthetic Frame")
        plt.axis("off")
        plt.show()

    def generate_pca_data(self, key, n):
        return jax.jit(sim_pca, static_argnums=(1, 2))(key, n, self.num_points)


class Detector(nn.Module):
    n_components: int
    n_suggestions: int
    latent_space_dim: int
    nframes: int = 11
    temporal_window: int = 3

    @nn.compact
    def __call__(self, D, is_training, A, B):
        batch_size, height, width, _ = D.shape
        npoints = self.temporal_window * (self.n_components + 2) + 1

        # Backbone feature extraction
        init_channels = 64 + sum([self.nframes // 5 * 2**i for i in range(6)])
        z = ResNet((2, 4, 4, 2), init_channels)(D, is_training)

        # Predictions: eigenvalues + scores
        w = nn.Dense(512)(jax.nn.relu(z))
        w = nn.BatchNorm(momentum=0.9, epsilon=1e-5)(w, use_running_average=not is_training)
        w = nn.Dense(self.n_suggestions * npoints)(w)
        w = w.reshape(batch_size, *w.shape[1:3], self.n_suggestions, npoints)
        w, scores = w[..., :-1], jax.nn.sigmoid(w[..., -1])

        # Adjust positions
        nrows, ncols = w.shape[1], w.shape[2]
        offsets = self.compute_offsets((height, width), (nrows, ncols))
        w = w.reshape(*w.shape[:-1], self.temporal_window, 2 + self.n_components)
        w = w.at[..., :2].add(offsets)
        w = w.reshape(batch_size, -1, self.temporal_window, 2 + self.n_components)

        # Align eigenvalues and convert to coordinates
        w = self.align_eigenvalues(w, B)
        coords = self.eigenvalues_to_coords(w, A)

        # Latent space encoding
        latent_space = LatentSpaceEncoder(self.latent_space_dim)(w, B, is_training)
        return scores.reshape(batch_size, -1), coords, latent_space

    def compute_offsets(self, frame_shape, grid_shape):
        height, width = frame_shape
        nrows, ncols = grid_shape
        y = jnp.linspace(height / nrows / 2, height - height / nrows / 2, nrows)
        x = jnp.linspace(width / ncols / 2, width - width / ncols / 2, ncols)
        return jnp.stack(jnp.meshgrid(x, y), axis=-1)[None, ..., None, None, :]

    def align_eigenvalues(self, w, B):
        eigenvalues = w[..., 2:]
        flipped = jnp.matmul(eigenvalues, B)
        ref = eigenvalues[..., 1:2, :]
        keep_dist = jnp.mean((ref - eigenvalues) ** 2, axis=-1, keepdims=True)
        flip_dist = jnp.mean((ref - flipped) ** 2, axis=-1, keepdims=True)
        aligned = jnp.where(keep_dist > flip_dist, flipped, eigenvalues)
        return w.at[..., 2:].set(aligned)

    def eigenvalues_to_coords(self, w, A):
        center_of_mass = w[..., :2]
        eigenvals = w[..., 2:]
        coords = eigenvals @ A
        return coords.reshape(*coords.shape[:-1], -1, 2) + center_of_mass[..., None, :]


class LatentSpaceEncoder(nn.Module):
    latent_dim: int

    def setup(self):
        bn_config = {"momentum": 0.9, "epsilon": 1e-5, "use_scale": True, "use_bias": True}
        self.fc_p1 = nn.Dense(128)
        self.bn_p1 = nn.BatchNorm(**bn_config)
        self.fc_p2 = nn.Dense(self.latent_dim)

    def __call__(self, x, B, is_training):
        x = jax.lax.stop_gradient(x)
        xf = x.at[..., 2:].set(jnp.matmul(x[..., 2:], B))

        p = jax.nn.relu(self.fc_p1(x.reshape(*x.shape[:2], -1)))
        pf = jax.nn.relu(self.fc_p1(xf.reshape(*x.shape[:2], -1)))
        p = self.bn_p1(p + pf, use_running_average=not is_training)
        p = self.fc_p2(p)
        return p


class ResNet(nn.Module):
    blocks_per_group: Sequence[int]
    init_channels: int
    bottleneck: bool = False

    @nn.compact
    def __call__(self, x, is_training):
        # BatchNorm configuration
        bn_config = {"momentum": 0.9, "epsilon": 1e-5, "use_scale": True, "use_bias": True}

        x = nn.Conv(self.init_channels, kernel_size=(7, 7), strides=(2, 2), use_bias=False)(x)
        x = nn.BatchNorm(**bn_config)(x, use_running_average=not is_training)
        x = jax.nn.relu(x)
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        for i, stride in enumerate((1, 2, 1, 2)):
            x = BlockGroup(
                channels=2 ** (6 + i),
                num_blocks=self.blocks_per_group[i],
                stride=stride,
                bn_config=bn_config,
                bottleneck=self.bottleneck,
            )(x, is_training)

        return x


class BlockGroup(nn.Module):
    channels: int
    num_blocks: int
    stride: int
    bn_config: dict
    bottleneck: bool = False

    @nn.compact
    def __call__(self, x, is_training):
        for i in range(self.num_blocks):
            x = ResNetBlock(
                channels=self.channels,
                stride=(self.stride if i == 0 else 1),
                bn_config=self.bn_config,
            )(x, is_training)
        return x


class ResNetBlock(nn.Module):
    channels: int
    stride: int
    bn_config: dict

    @nn.compact
    def __call__(self, x, is_training):
        residual = x
        x = nn.Conv(self.channels, kernel_size=(3, 3), strides=(self.stride, self.stride))(x)
        x = nn.BatchNorm(**self.bn_config)(x, use_running_average=not is_training)
        x = jax.nn.relu(x)

        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(**self.bn_config)(x, use_running_average=not is_training)

        if residual.shape != x.shape:
            residual = nn.Conv(
                self.channels, kernel_size=(1, 1), strides=(self.stride, self.stride)
            )(residual)

        return jax.nn.relu(x + residual)


def _importance_weights(n):
    weights = 1 / (jnp.abs(jnp.arange(-n // 2 + 1, n // 2 + 1)) + 1)
    return weights / weights.sum()


def calc_losses(predictions, labels, sigma=1.0, cutoff=1.0, size=256):
    S_pred, X_pred, P_pred = predictions

    inside = jnp.all((labels >= 0) & (labels < size), axis=(-1, -2, -3))

    @jax.vmap
    def distace_matrix(a, b):
        A = a[None, ...]
        B = b[:, None, ...]
        return jnp.sum((A - B) ** 2, axis=-1)

    # Compute the distance matrix for direct and flip versions
    distance = distace_matrix(X_pred, labels).mean(-1)
    flip_distance = distace_matrix(X_pred, jnp.flip(labels, axis=-2)).mean(-1)
    distances = jnp.minimum(distance, flip_distance)

    # Reduce the distance to be weighted by the importance of each frame.
    num_frames = X_pred.shape[2]
    temporal_weights = _importance_weights(num_frames)
    distances = jnp.average(distances, axis=-1, weights=temporal_weights)

    # Compute the loss of the points only taking into consideration only those
    # predictions that are inside.
    inside_count = jnp.sum(inside) + 1e-6
    masked_distances = distances * inside[:, :, None]
    Loss_X = jnp.sum(jnp.min(masked_distances, axis=2)) / inside_count

    # Before computing the score and latent space losses,
    # we stop gradients for of the distances.
    distances = jax.lax.stop_gradient(distances)
    X = jax.lax.stop_gradient(X_pred)

    # Compute the confidence score of each prediction as S = exp(-d2/sigma)
    # and perform L2 loss.
    scores = jnp.exp(-jnp.min(distances, axis=1) / sigma)
    Loss_S = jnp.mean((scores - S_pred) ** 2)

    # Find out which target is closests to each prediction.
    # ASSUMPTION: That is the one they are predicting.
    T = jnp.argmin(distances, axis=1)

    # Compute which permutations are targeting the same index on a matrix.
    # T(i,j) = T(j, i) = 1 if i,j 'target' the same label, 0 otherwise
    same_T = T[:, None, :] == T[:, :, None]

    # Visibility mask for far predictions that not should not share latent
    # space.
    distance_ls = distace_matrix(P_pred, P_pred)
    K = X.shape[3]
    Xcm = X[:, :, num_frames // 2, K // 2, :]  # [B N Wt K 2]
    visible = distace_matrix(Xcm, Xcm) < cutoff**2
    factor = visible / visible.sum(axis=2)[:, :, None]

    # Compute the cross entropy loss depending on whether they aim to predict
    # the same target. P(i targets k| j targets k) ~= e^(-d^2)
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    safe_log = lambda x: jnp.log(jnp.where(x > 0.0, x, 1.0)) # pyright: ignore
    atraction = distance_ls  # log(exp(d2))
    repulsion = -safe_log(1 - jnp.exp(-distance_ls))
    Loss_P = factor * jnp.where(same_T, atraction, repulsion)

    # Only take into account the predictions that are actually preddicting.
    # Bad prediction should not be close to actual predictions in the latent
    # space.
    scores_matrix = scores[:, :, None] * scores[:, None, :]
    Loss_P = jnp.sum(scores_matrix * Loss_P) / (scores_matrix.sum() + 1e-6)
    return Loss_X, Loss_S, Loss_P


if __name__ == "__main__":
    key = jax.random.key(42)

    dataset = CElegansDataset(
        num_worms=50,
        num_points=49,
        num_frames=11,
        batch_size=8,
        box_size=256,
        video_size=256,
    )

    get_batch = jax.jit(lambda key: dataset.get_batch(key))
    X, y = get_batch(key)


    model = Detector(12, n_suggestions=8, latent_space_dim=8)
    pca_data = dataset.generate_pca_data(key, 100_000)
    pca_state = pcax.fit(pca_data.reshape(len(pca_data), -1), 12)
    A = pca_state.components
    num_components, kpoints2 = A.shape
    J = jnp.flip(jnp.identity(kpoints2), axis=1)
    B = A @ J @ A.T

    variables = model.init(key, X, is_training=False, A=A, B=B)
    params, batch_stats = variables["params"], variables["batch_stats"]
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, X, y, opt_state, batch_stats):
        def loss_fn(params):
            variables = {"params": params, "batch_stats": batch_stats}
            outs, updates = model.apply(
                variables, X, is_training=True, A=A, B=B, mutable="batch_stats"
            )
            (loss_x, loss_s, loss_p) = calc_losses(outs, y)
            loss = 1e0 * loss_x + 1e2 * loss_s + 1e5 * loss_p
            losses = {'x': loss_x, 's': loss_s, 'p': loss_p}
            return loss, (updates["batch_stats"], losses)

        (loss, (batch_stats, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state, batch_stats, losses

    for step in (bar := tqdm(range(int(1e4)), ncols=120)):
        data_key = jax.random.fold_in(key, step)
        X, y = get_batch(data_key)
        loss, params, opt_state, batch_stats, losses = train_step(params, X, y, opt_state, batch_stats)
        bar.set_description(f"Loss: {loss:.2g}")
