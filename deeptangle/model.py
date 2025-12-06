"""
DeepTangle model using Flax NNX.

Implements ResNet-based worm detection with latent space encoding.
"""
from typing import Sequence
from flax import nnx
import jax
import jax.numpy as jnp


class ResNetBlock(nnx.Module):
    """Basic ResNet block."""
    
    def __init__(self, in_channels: int, channels: int, stride: int = 1, *, rngs: nnx.Rngs):
        self.channels = channels
        self.stride = stride
        self.in_channels = in_channels
        
        # First conv path
        self.conv1 = nnx.Conv(
            in_channels, channels, kernel_size=(3, 3),
            strides=(stride, stride), padding='SAME',
            use_bias=False, rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
        
        # Second conv path
        self.conv2 = nnx.Conv(
            channels, channels, kernel_size=(3, 3),
            padding='SAME', use_bias=False, rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(channels, rngs=rngs)
        
        # Projection for residual if needed
        self.needs_projection = (stride != 1 or in_channels != channels)
        if self.needs_projection:
            self.proj_conv = nnx.Conv(
                in_channels, channels, kernel_size=(1, 1),
                strides=(stride, stride), use_bias=False, rngs=rngs
            )
            self.proj_bn = nnx.BatchNorm(channels, rngs=rngs)
        else:
            self.proj_conv = None
            self.proj_bn = None
    
    def __call__(self, x, *, train: bool = True):
        residual = x
        
        # First conv
        y = self.conv1(x)
        y = self.bn1(y, use_running_average=not train)
        y = nnx.relu(y)
        
        # Second conv
        y = self.conv2(y)
        y = self.bn2(y, use_running_average=not train)
        
        # Adjust residual dimensions if needed
        if self.needs_projection:
            residual = self.proj_conv(x)
            residual = self.proj_bn(residual, use_running_average=not train)
        
        return nnx.relu(y + residual)


class ResNet(nnx.Module):
    """ResNet backbone for feature extraction."""
    
    def __init__(
        self,
        in_channels: int,
        init_channels: int = 64,
        blocks_per_group: Sequence[int] = (2, 4, 4, 2),
        channels_per_group: Sequence[int] = (64, 128, 256, 512),
        strides: Sequence[int] = (1, 2, 1, 2),
        *,
        rngs: nnx.Rngs
    ):
        self.init_conv = nnx.Conv(
            in_channels, init_channels, kernel_size=(7, 7),
            strides=(2, 2), padding='SAME', use_bias=False, rngs=rngs
        )
        self.init_bn = nnx.BatchNorm(init_channels, rngs=rngs)
        
        # Create ResNet blocks
        self.block_groups = nnx.List([])
        current_channels = init_channels
        for i, (num_blocks, channels, stride) in enumerate(zip(
            blocks_per_group, channels_per_group, strides
        )):
            blocks = []
            for j in range(num_blocks):
                block_stride = stride if j == 0 else 1
                block_in_channels = current_channels if j == 0 else channels
                blocks.append(ResNetBlock(block_in_channels, channels, stride=block_stride, rngs=rngs))
                current_channels = channels
            self.block_groups.append(nnx.List(blocks))
    
    def __call__(self, x, *, train: bool = True):
        # Initial conv
        x = self.init_conv(x)
        x = self.init_bn(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        
        # ResNet blocks
        for blocks in self.block_groups:
            for block in blocks:
                x = block(x, train=train)
        
        return x


class LatentSpaceEncoder(nnx.Module):
    """Latent space encoder with orientational invariance."""
    
    def __init__(self, latent_dim: int, input_dim: int, *, rngs: nnx.Rngs):
        self.latent_dim = latent_dim
        self.fc1 = nnx.Linear(input_dim, 128, rngs=rngs)
        self.bn = nnx.BatchNorm(128, rngs=rngs)
        self.fc2 = nnx.Linear(128, latent_dim, rngs=rngs)
    
    def __call__(self, x, B, *, train: bool = True):
        # Stop gradients as in original implementation
        x = jax.lax.stop_gradient(x)
        
        # Create flipped version
        xf = x.at[..., 2:].set(jnp.matmul(x[..., 2:], B))
        
        # Process both versions
        x_flat = x.reshape(*x.shape[:2], -1)
        xf_flat = xf.reshape(*xf.shape[:2], -1)
        
        p = nnx.relu(self.fc1(x_flat))
        pf = nnx.relu(self.fc1(xf_flat))
        
        # Combine and batch norm
        p = self.bn(p + pf, use_running_average=not train)
        p = self.fc2(p)
        
        return p


class Detector(nnx.Module):
    """
    Main detection network combining CNN backbone and latent space encoder.
    
    This uses Flax NNX for better Hugging Face Hub compatibility.
    """
    
    def __init__(
        self,
        npoints: int,
        n_suggestions: int,
        latent_dim: int,
        nframes: int = 11,
        *,
        rngs: nnx.Rngs
    ):
        self.npoints = npoints
        self.n_suggestions = n_suggestions
        self.latent_dim = latent_dim
        self.nframes = nframes
        
        self.neigen = npoints
        self.temporal_window = 3
        self.npoints_output = self.temporal_window * (npoints + 2) + 1
        
        # Feature extraction
        init_channels = 64 + sum([nframes // 5 * 2**i for i in range(6)])
        self.backbone = ResNet(
            in_channels=nframes,  # Number of time frames
            init_channels=init_channels,
            blocks_per_group=(2, 4, 4, 2),
            channels_per_group=(64, 128, 256, 512),
            strides=(1, 2, 1, 2),
            rngs=rngs
        )
        
        # Position prediction layers
        # Input dimension from backbone will be channels_per_group[-1]
        self.fc_w1 = nnx.Linear(512, 512, rngs=rngs)  # Last channel group is 512
        self.bn_w = nnx.BatchNorm(512, rngs=rngs)
        self.fc_w2 = nnx.Linear(512, n_suggestions * self.npoints_output, rngs=rngs)
        
        # Latent space encoder
        # Input dimension is temporal_window * (2 + npoints)
        encoder_input_dim = self.temporal_window * (2 + npoints)
        self.encoder = LatentSpaceEncoder(latent_dim, encoder_input_dim, rngs=rngs)
    
    def __call__(self, D, B, *, train: bool = True):
        batch_size, height, width, channels = D.shape
        
        # Extract features
        z = self.backbone(D, train=train)
        
        # Predict positions (eigenvalues + center of mass + score)
        w = nnx.relu(self.fc_w1(z))
        w = self.bn_w(w, use_running_average=not train)
        w = self.fc_w2(w)
        
        nrows, ncols = w.shape[1], w.shape[2]
        w = w.reshape(batch_size, nrows, ncols, self.n_suggestions, self.npoints_output)
        
        # Create offset grid for converting grid positions to pixel positions
        s_y = (height / nrows / 2) + jnp.arange(0, height, height / nrows)
        s_x = width / ncols / 2 + jnp.arange(0, width, width / ncols)
        offset = jnp.stack(jnp.meshgrid(s_x, s_y), axis=-1)
        
        # Separate score prediction from positional
        w, s = w[..., :-1], w[..., -1]
        
        # Reshape to temporal window format
        w = w.reshape(*w.shape[:-1], self.temporal_window, 2 + self.neigen)
        
        # Add offset to convert grid positions to pixel coordinates
        w = w.at[..., (0, 1)].add(offset[None, ..., None, None, :])
        
        # Flatten the prediction from grid to 1D
        w = w.reshape(batch_size, -1, self.temporal_window, 2 + self.neigen)
        s = s.reshape(batch_size, -1)
        
        # Align eigenvalues (handle flipping)
        w = self.align_eigenvalues(w, B)
        
        # Predict latent space
        p = self.encoder(w, B, train=train)
        
        return s, w, p
    
    def align_eigenvalues(self, x, B):
        """Align eigenvalues to handle head-tail ambiguity."""
        w = x[..., 2:]
        wf = jnp.matmul(w, B)
        dist_keep = jnp.mean((w[..., 1:2, :] - w) ** 2, axis=-1, keepdims=True)
        dist_flip = jnp.mean((w[..., 1:2, :] - wf) ** 2, axis=-1, keepdims=True)
        w = jnp.where(dist_keep > dist_flip, wf, w)
        x = x.at[..., 2:].set(w)
        return x


def create_detector(
    npoints: int,
    n_suggestions: int,
    latent_dim: int,
    nframes: int = 11,
    *,
    rngs: nnx.Rngs
) -> Detector:
    """
    Create a Detector model using Flax NNX.
    
    Args:
        npoints: Number of PCA components.
        n_suggestions: Number of suggestions per grid cell.
        latent_dim: Dimension of latent space.
        nframes: Number of frames in input clip.
        rngs: Random number generators for initialization.
        
    Returns:
        A Detector model instance.
    """
    return Detector(
        npoints=npoints,
        n_suggestions=n_suggestions,
        latent_dim=latent_dim,
        nframes=nframes,
        rngs=rngs
    )
