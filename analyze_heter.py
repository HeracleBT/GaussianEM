import numpy as np
import sys
from utils.plot_utils import show_gaussians
from utils.general_utils import t2a
from model.render_query_EM import query, render
from argparse import ArgumentParser
import json
import torch
from model.gaussian_model_insight import GaussianModel, SimpleGaussian
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from utils.eval_em import cluster_kmeans, get_nearest_point, run_umap, run_pca, get_pc_traj
from scipy.spatial import cKDTree
import open3d as o3d
from sklearn.metrics import pairwise_distances
from model.gmmvae_heter import HetGMMVAESimpleIndependent
from torch import nn
import mrcfile
import seaborn as sns
import os


def generate_vol(model, zlists, gaussians, gaussians_info, queryfunc, output_dir, mask_ind=None):
    N = gaussians._density.shape[0]
    base_density = gaussians_info[..., :1]
    base_scaling = gaussians_info[..., 1:2]
    base_xyz = gaussians_info[..., 2:]
    for i in range(len(zlists)):
        z_value = zlists[i]
        eval_z = torch.tensor(z_value, device="cuda", dtype=torch.float).view(1, -1)
        gaussian_param = model.decode(eval_z, gaussians_info)

        gaussian_param_xyz = gaussian_param[2].view(N, 3)
        gaussian_param_density = gaussian_param[0].view(N, 1)
        gaussian_param_scaling = gaussian_param[1].view(N, 1)

        gaussians._density = base_density + gaussian_param_density
        gaussians._scaling = base_scaling
        if mask_ind is not None:
            gaussians._xyz = base_xyz + gaussian_param_xyz *  mask_ind
        else:
            gaussians._xyz = base_xyz + gaussian_param_xyz

        vol_pred = queryfunc(gaussians)["vol"]
        vol_pred_mrc = "%s/vol_%d.mrc" % (output_dir, i)

        with mrcfile.new(vol_pred_mrc, overwrite=True) as mrc:
            mrc.set_data(t2a(vol_pred).astype(np.float32))
            mrc.voxel_size = pixel_size


if __name__ == "__main__":

    parser = ArgumentParser(description="Analyzing")
    parser.add_argument("--source_dir", type=str, default=None)
    parser.add_argument("--gaussians", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--model_weight", type=str, default=None)
    parser.add_argument("--latent", type=str, default=None)
    parser.add_argument("--nclass", type=int, default=5)
    parser.add_argument("--mask", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    source_dir = args.source_dir
    pc_path = "%s/%s" % (source_dir, args.gaussians)
    cfg_path = "%s/%s" % (source_dir, args.cfg)
    with open(cfg_path, "r") as f:
        meta_data = json.load(f)
    scanner_cfg = meta_data['cfg']
    pixel_size = scanner_cfg["pixelSize"]
    vol_size = scanner_cfg['nVoxel']
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = np.array([scanner_cfg["scale_min"], scanner_cfg["scale_max"]]) * volume_to_world
    gaussians = GaussianModel(scale_bound, max_density=None)
    gaussians.load_ply(pc_path)

    if args.mask:
        mask_ind = np.load(args.mask)[:, None].astype(np.float32)
        mask_ind = torch.tensor(mask_ind).cuda()
    else:
        mask_ind = None

    D = vol_size[0]
    gaussian_embedding_dim = scanner_cfg['gaussian_embedding_dim']
    model = HetGMMVAESimpleIndependent(
        qlayers=scanner_cfg['qlayers'],
        qdim=scanner_cfg['qdim'],
        in_dim=D * D,
        zdim=scanner_cfg['zdim'],
        gaussian_embedding_dim=gaussian_embedding_dim,
        gaussian_kdim=scanner_cfg['gaussian_kdim'],
        gaussian_klayers=scanner_cfg['gaussian_klayers'],
        feature_dim=scanner_cfg['feature_dim'],
        feature_kdim=scanner_cfg['feature_kdim'],
        feature_klayers=scanner_cfg['feature_klayers'],
        activation=nn.ReLU,
        archi_type='MLP'
    )
    model.to(device="cuda")

    model_file = "%s/%s" % (source_dir, args.model_weight)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["model_state_dict"])

    z_mu_pth = "%s/%s" % (source_dir, args.latent)
    with open(z_mu_pth, 'rb') as f:
        model_output = pickle.load(f)
    z_mus = model_output["z_mu"]

    K = args.nclass
    kmeans_labels, centers = cluster_kmeans(z_mus, K)
    centers, centers_ind = get_nearest_point(z_mus, centers)

    gaussians1 = SimpleGaussian(scale_bound, ['density', 'scaling', 'position'], gaussians.max_density)
    gaussians1.create_from_Gaussian(gaussians)
    base_density = gaussians1._density.clone().detach()
    base_scaling = gaussians1._scaling.clone().detach()
    base_xyz = gaussians1._xyz.clone().detach()
    base_pos = gaussians1.get_xyz.clone().detach()
    base_info = torch.cat([base_density, base_scaling, base_pos], dim=-1)
    
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        reorder=True
    )
    kmeans_dir = "%s/kmeans_%d" % (source_dir, K)
    if not os.path.exists(kmeans_dir):
        os.mkdir(kmeans_dir)
    generate_vol(model, centers, gaussians1, base_info, queryfunc, kmeans_dir, mask_ind=mask_ind)


    pc, pca = run_pca(z_mus)
    
    for dim in [1, 2, 3]:
        lim = [10, 50, 90]
        pc_values = np.percentile(pc[:, dim - 1], lim)
        z_traj = get_pc_traj(pca, scanner_cfg['zdim'], len(lim), dim, None, None, pc_values)
        pca_dir = "%s/pc_%d" % (source_dir, dim)
        if not os.path.exists(pca_dir):
            os.mkdir(pca_dir)
        generate_vol(model, centers, gaussians1, base_info, queryfunc, pca_dir, mask_ind=mask_ind)

