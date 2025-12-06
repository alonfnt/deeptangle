# Split Training Workflow

This document describes the new modular training workflow that separates dataset generation from model training.

## Overview

The original `train.py` script combined simulation, dataset generation, and model training in a single process using Haiku. The new workflow splits these into separate steps:

1. **Dataset Generation**: Pre-generate synthetic training data
2. **Model Training**: Train the model using pre-generated data with Flax

This separation provides several benefits:
- Better modularity and maintainability
- Ability to reuse datasets across multiple training runs
- Easier experimentation with different model architectures
- Better HuggingFace Hub compatibility (Flax models)
- Reduced memory footprint during training

## Workflow

### Step 1: Generate Synthetic Dataset

Use `generate_dataset.py` to create a synthetic training dataset:

```bash
python generate_dataset.py \
    --num_samples=10000 \
    --num_val_samples=1000 \
    --nframes=11 \
    --size=256 \
    --nworms=5,10,50,100,150,200,250 \
    --kpoints=49 \
    --npca=12 \
    --nworms_pca=100000 \
    --output_dir=synthetic_dataset \
    --seed=42
```

**Key Arguments:**
- `--num_samples`: Number of training samples to generate
- `--num_val_samples`: Number of validation samples to generate
- `--nframes`: Number of frames per clip
- `--size`: Frame size (height and width)
- `--nworms`: Comma-separated list of worm counts to randomly sample from
- `--npca`: Number of PCA components
- `--output_dir`: Directory to save the generated dataset

**Output Structure:**
```
synthetic_dataset/
├── train/
│   ├── sample_000000.npz
│   ├── sample_000001.npz
│   └── ...
├── val/
│   ├── sample_000000.npz
│   ├── sample_000001.npz
│   └── ...
├── pca_matrix.npy
└── metadata.json
```

### Step 2: Train Model with Pre-Generated Dataset

Use `train_from_dataset.py` to train the model:

```bash
python train_from_dataset.py \
    --dataset_dir=synthetic_dataset \
    --batch_size=32 \
    --train_steps=100000 \
    --eval_interval=100 \
    --n_suggestions=8 \
    --latent_dim=8 \
    --learning_rate=0.001 \
    --checkpoint_dir=checkpoints_flax \
    --save
```

**Key Arguments:**
- `--dataset_dir`: Path to the pre-generated dataset directory
- `--batch_size`: Training batch size
- `--train_steps`: Number of training steps
- `--n_suggestions`: Number of suggestions per grid cell
- `--latent_dim`: Dimension of latent space
- `--save`: Enable checkpoint saving

## Model Architecture

The new workflow uses a Flax-based model (`deeptangle/model_flax.py`) instead of the original Haiku model. The architecture is functionally equivalent but implemented in Flax for better HuggingFace compatibility.

### Key Components:
- **ResNet backbone**: Feature extraction from input clips
- **Detector head**: Predicts worm positions and scores
- **Latent space encoder**: Learns orientational-invariant representations

## Comparison with Original Workflow

| Aspect | Original (`train.py`) | New Workflow |
|--------|---------------------|--------------|
| Dataset | Generated on-the-fly | Pre-generated |
| Framework | Haiku | Flax |
| Memory | Higher (simulation + training) | Lower (training only) |
| Flexibility | Coupled | Modular |
| HuggingFace | Limited support | Better support |

## Data Loader

The `deeptangle.dataset.loader` module provides utilities for loading pre-generated datasets:

```python
from deeptangle.dataset.loader import SyntheticDatasetLoader

# Load training data
loader = SyntheticDatasetLoader("synthetic_dataset", split="train")
print(f"Dataset size: {len(loader)}")

# Get a single sample
X, y = loader[0]

# Create batch iterator
batch_iter = loader.get_batch_iterator(batch_size=32, shuffle=True)
batch_X, batch_y = next(batch_iter)
```

## Testing

To quickly test the pipeline with small parameters:

```bash
# Generate small test dataset
python generate_dataset.py \
    --num_samples=100 \
    --num_val_samples=20 \
    --nframes=5 \
    --size=64 \
    --nworms=5,10 \
    --kpoints=11 \
    --npca=3 \
    --nworms_pca=100 \
    --output_dir=test_dataset

# Train with test dataset
python train_from_dataset.py \
    --dataset_dir=test_dataset \
    --batch_size=4 \
    --train_steps=100 \
    --eval_interval=10 \
    --n_suggestions=4 \
    --latent_dim=2
```

## Future Improvements

- [ ] Add checkpoint saving and loading for Flax models
- [ ] Implement HuggingFace Hub integration
- [ ] Add data augmentation options for pre-generated datasets
- [ ] Create conversion script from Haiku to Flax weights
- [ ] Add TensorBoard logging
- [ ] Implement distributed training support

## Notes

- The original `train.py` script remains available for backward compatibility
- Both Haiku (`dm-haiku`) and Flax are included as dependencies
- The PCA matrix is computed once during dataset generation and reused for training
