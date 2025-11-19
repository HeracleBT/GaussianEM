# GaussianEM

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

### Preprocess particle stack, poses, ctf information

Extract 2D particles images, orientations and contrast transfer function (ctf) parameters of particles from .star file (RELION format)

### Train a heterogeneity network

The GaussianEM can be trained with following command:

```
Usage: python train_em_relion_vae_xyz_batch.py -s [DATA DIRECTORY] --particle_name [PARTICLE STACK NAME] --pose [POSES FILE] --ctf [CTF FILE] --point [GAUSSIAN FILE] --cfg [CONFIGURE FILE] --output [OUTPUT DIRECTORY] --batch [BATCH SIZE]

---options---
-s [str] : directory of data
--particle_name [str] : particle stacks (.mrcs)
--pose [str] : orientations of particles (.npy)
--ctf [str] : contrast transfer function (ctf) information of particles (.npy)
--point [str] : attributes of 3D Gaussians (.ply)
--cfg [str] : hyperparameters for Gaussians and heterogeneity network (.json)
--output [str] : directory of ouput
--batch [int] : batch size of inputted images during training
```

Example command to train a heterogeneity network for 20 epochs on the IgG-1D dataset

```
Example: python train_em_relion_vae_xyz_batch.py -s dataset/IgG-1D \
    --particle_name img_stack \
    --pose poses.npy \
    --ctf ctf.npy \
    --point point_cloud.ply \
    --cfg cfg_scale_0.5_1.0.json \
    --output heter_density_scaling_pos_zdim10_dp \
    --batch 32
```

The datasets EMPAIR-(10059, 10180, 10343, 10345, 10841, T6SS_effectors) are available via https://drive.google.com/drive/folders/1185kjegtrHnsF0R1N7DUXD04UjcqxFxw?usp=sharing

### Analyze the results

This script provides a series of standard analysis:

* Generation of volumes after K-means clustering
* Generation of trajectories along the first, second and third principal components of the latent embeddings 


```
Usage: python analyze_heter.py --source_dir [DATA DIRECTORY] --gaussians [GAUSSIAN FILE] --cfg [CONFIGURE FILE] --model_weight [MODEL] --latent [LATENT VIRIABLE] --nclass [NUMBER OF CLASS]

---options---
--source_dir [str] : directory of data
--gaussians [str] : attributes of 3D Gaussians (.ply)
--cfg [str] : hyperparameters for Gaussians and heterogeneity network (.json)
--model_weight [str] : trained parameters of heterogeneity network (.pkl)
--latent [str] : latent viriables of particles extracted from tranined network (.pkl)
--nclass [int] : number of discrete class for K-means clustering
```


Example command to analyze the result of IgG-1D

```
Example: python analyze_heter.py --source_dir dataset/IgG-1D \
    --gaussians point_cloud.ply \
    --cfg cfg.json \
    --model_weight weights.19.pkl \
    --latent weights.19.output.pkl \
    --nclass 10
```