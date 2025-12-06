"""
Flax-based implementation of the DeepTangle model.

This is a more HuggingFace-compatible version that doesn't depend on Haiku
and can work with pre-generated datasets.
"""
from typing import Optional, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp


class ResNetBlock(nn.Module):
    """Basic ResNet block."""
    channels: int
    stride: int = 1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        
        # First conv
        y = nn.Conv(self.channels, (3, 3), strides=(self.stride, self.stride), 
                    padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        
        # Second conv
        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        
        # Adjust residual dimensions if needed
        if self.stride != 1 or x.shape[-1] != self.channels:
            residual = nn.Conv(self.channels, (1, 1), strides=(self.stride, self.stride), 
                              use_bias=False)(x)
            residual = nn.BatchNorm(use_running_average=not train)(residual)
        
        return nn.relu(y + residual)


class ResNet(nn.Module):
    """ResNet backbone for feature extraction."""
    blocks_per_group: Sequence[int] = (2, 4, 4, 2)
    channels_per_group: Sequence[int] = (64, 128, 256, 512)
    strides: Sequence[int] = (1, 2, 1, 2)
    init_channels: int = 64
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Initial conv
        x = nn.Conv(self.init_channels, (7, 7), strides=(2, 2), 
                    padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        
        # ResNet blocks
        for i, (num_blocks, channels, stride) in enumerate(zip(
            self.blocks_per_group, self.channels_per_group, self.strides
        )):
            for j in range(num_blocks):
                block_stride = stride if j == 0 else 1
                x = ResNetBlock(channels, stride=block_stride)(x, train)
        
        return x


class LatentSpaceEncoder(nn.Module):
    """Latent space encoder with orientational invariance."""
    latent_dim: int
    
    @nn.compact
    def __call__(self, x, B, train: bool = True):
        # Stop gradients as in original implementation
        x = jax.lax.stop_gradient(x)
        
        # Create flipped version
        xf = x.at[..., 2:].set(jnp.matmul(x[..., 2:], B))
        
        # Process both versions
        p = nn.Dense(128)(x.reshape(*x.shape[:2], -1))
        p = nn.relu(p)
        
        pf = nn.Dense(128)(xf.reshape(*x.shape[:2], -1))
        pf = nn.relu(pf)
        
        # Combine and batch norm
        p = nn.BatchNorm(use_running_average=not train)(p + pf)
        p = nn.Dense(self.latent_dim)(p)
        
        return p


class Detector(nn.Module):
    """
    Main detection network combining CNN backbone and latent space encoder.
    
    This is a Flax version of the original Haiku model, designed to be
    more compatible with Hugging Face Hub.
    """
    npoints: int
    n_suggestions: int
    latent_dim: int
    nframes: int = 11
    
    def setup(self):
        self.neigen = self.npoints
        self.temporal_window = 3
        self.npoints_output = self.temporal_window * (self.npoints + 2) + 1
        
        # Feature extraction
        init_channels = 64 + sum([self.nframes // 5 * 2**i for i in range(6)])
        self.backbone = ResNet(
            blocks_per_group=(2, 4, 4, 2),
            channels_per_group=(64, 128, 256, 512),
            init_channels=init_channels
        )
        
        # Position prediction layers
        self.fc_w1 = nn.Dense(512)
        self.fc_w2 = nn.Dense(self.n_suggestions * self.npoints_output)
        
        # Latent space encoder
        self.encoder = LatentSpaceEncoder(self.latent_dim)
    
    @nn.compact
    def __call__(self, D, B, train: bool = True):
        batch_size, height, width, channels = D.shape
        
        # Extract features
        z = self.backbone(D, train)
        
        # Predict positions (eigenvalues + center of mass + score)
        w = nn.relu(self.fc_w1(z))
        w = nn.BatchNorm(use_running_average=not train)(w)
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
        p = self.encoder(w, B, train)
        
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
    nframes: int = 11
) -> Detector:
    """
    Create a Detector model.
    
    Args:
        npoints: Number of PCA components.
        n_suggestions: Number of suggestions per grid cell.
        latent_dim: Dimension of latent space.
        nframes: Number of frames in input clip.
        
    Returns:
        A Detector model instance.
    """
    return Detector(
        npoints=npoints,
        n_suggestions=n_suggestions,
        latent_dim=latent_dim,
        nframes=nframes
    )
