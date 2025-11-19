# CryoAlign

Here is an official implementation of GaussianEM, a method for modelling compositional and conformational heterogeneity using 3D Gaussians.

Understanding protein flexibility and its dynamic interactions with other molecules is essential for protein function study. Cryogenic electron microscopy (cryo-EM) provides an opportunity to directly observe macromolecular dynamics. However, analyzing datasets that contain both continuous motions and discrete states remains highly challenging. Here we present GaussianEM, a Gaussian pseudo-atomic framework that simultaneously models compositional and conformational heterogeneity from experimental cryo-EM images. GaussianEM employs a two-encoder-one-decoder architecture to map an image to its individual Gaussian components, and represent structural variability through changes in Gaussian parameters. This approach provides an intuitive and interpretable description of conformational changes, preserves local structural consistency along the transition trajectories, and naturally bridges the gap between density-based models and corresponding atomic models. We demonstrate the effectiveness of GaussianEM on both simulated and experimental datasets.

## Operation System

Ubuntu 18.04 or later

## Requirements

GaussianEM is compatible with Python version 3.7; we recommend using a clean conda environment.

### Python environment:

base environment:
```
conda create -n gaussian_em python=3.7
conda activate gaussian_em
pip install -r requirements.txt
```

additional packages for image rendering:
```
pip install -e submodules/simple-knn
pip install -e submodules/xray-gaussian-rasterization-voxelization
```

## Usage


