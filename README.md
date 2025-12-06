# de(ep)tangle

[![Communications Biology](https://img.shields.io/badge/CommsBio-10.1038/s42003--023--05098--1-blue?style=flat)](https://www.nature.com/articles/s42003-023-05098-1)
[![arXiv](https://img.shields.io/badge/arXiv-2301.04460-b31b1b.svg?style=flat)](https://arxiv.org/abs/2301.04460)

This repository contains the implementation of [Fast detection of slender bodies in high density microscopy data](https://www.nature.com/articles/s42003-023-05098-1) paper.

<p align="center">
  <img src="https://github.com/kirkegaardlab/deeptanglelabel/blob/main/docs/figures/tracking.gif" height="236" />
  <img src="https://github.com/kirkegaardlab/deeptanglelabel/blob/main/docs/figures/following.gif" height="236" />
  <img src="https://github.com/kirkegaardlab/deeptanglelabel/blob/main/docs/figures/dense.png" width="720" />
</p>

## Installation
To run the code one must first install the dependencies.
You can do this in a virtual environment:

```setup
python3 -m venv venv
source venv/bin/activate
```

Start by installing `jax` following instructions at their [repository](https://github.com/google/jax?tab=readme-ov-file#installation).
Install the remaining dependencies afterwards:

```setup
pip install -r requirements.txt
```

If you need to use the model and the auxiliary functions outside this repository, you can install it from the root folder by
```install
pip install -e .
```

## Train
To train the model, there are two approaches:

### New Modular Workflow (Recommended for HuggingFace)
The new workflow separates dataset generation from training:

1. **Generate synthetic dataset:**
```train
python3 generate_dataset.py --num_samples=10000 --num_val_samples=1000 --nframes=11 --size=256 --nworms=5,10,50,100,150,200,250 --output_dir=synthetic_dataset
```

2. **Train with pre-generated dataset:**
```train
python3 train_from_dataset.py --dataset_dir=synthetic_dataset --batch_size=32 --train_steps=100000 --save
```

See [WORKFLOW.md](./WORKFLOW.md) for detailed documentation.

### Original On-the-fly Training
The original training script generates data during training:
```train
python3 train.py --batch_size=32 --eval_interval=10 --nworms=100,200 --save
```

The possible arguments can be seen by using the help flag.
```train
python3 train.py --help
```

An example of a training run would be
```train
python3 train.py --batch_size=32 --eval_interval=10 --nworms=100,200 --save
```

## Usage
Example scripts such as detection and tracking can be found in the [examples folder](./examples)

We include a Dockerfile (cpu only). For linux we provide a script to run the relevant commands:
```
(sudo) sh docker_run.sh
```


## Weights
The weights used in the paper can be downloaded from [here](https://sid.erda.dk/share_redirect/cEjIpG1yQl)
or by using the following commmand
```download
wget https://sid.erda.dk/share_redirect/cEjIpG1yQl -O weights.zip
```

