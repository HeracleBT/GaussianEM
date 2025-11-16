import os
import os.path as osp
import torch
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import mrcfile
import starfile
import json
from collections import defaultdict
import torch.nn.functional as F
import pickle

sys.path.append("./")
from arguments import ModelParams_EM, OptimizationParams_EM, PipelineParams
from model.gaussian_model_insight import GaussianModel, SimpleGaussian
from model.render_query_EM import query, render
from utils.general_utils import safe_state
from utils.general_utils import load_config
from utils.general_utils import t2a
from utils.utils_em import circular_mask, R_from_relion, relion_angle_to_matrix
from data.dataset_em import ImageDataset, make_dataloader
from particle_preprocess import compute_single_ctf, ctf_params
from model.gmmvae_heter import HetGMMVAESimpleIndependent
from torch import nn
from utils.general_utils import mkdir_p
from scipy.spatial import cKDTree
from sklearn.neighbors import radius_neighbors_graph
from utils.eval_em import cluster_kmeans, get_nearest_point

epsilon = 0.0000005

class Particles:
    gaussians: GaussianModel

    def __init__(self, dataset_dir, particle_name, half_name, pose_file, ctf_file, point_cloud, cfg, window=True):
        self.path = dataset_dir
        self.particle_name = particle_name
        self.half_name = half_name
        self.images_dir = "%s/%s.mrcs" % (self.path, half_name)
        self.orientation_dir = "%s/%s" % (self.path, pose_file)
        self.vol_dir = "%s/%s.mrc" % (self.path, particle_name)
        self.cfg_file = "%s/%s" % (self.path, cfg)
        
        self.init_cfg()
        invert_data = True if self.scanner_cfg['invert_contrast'] else False
        data = ImageDataset(
            mrcfile=self.images_dir,
            lazy=True,
            invert_data=invert_data,
            ind=None,
            window=window,
            datadir=None,
        )
        self.particle_num = data.N
        self.images = data
        self.particle_size = data.D
        self.size_scale = self.particle_size / 2  ### scale vol into [-1, 1], stablize training

        self.ctf_dir = "%s/%s" % (self.path, ctf_file)
        self.ctf_param = np.load(self.ctf_dir)
        self.ctf_mask = circular_mask(self.particle_size).cuda()

        self.load_poses()        
        self.point_dir = "%s/%s" % (self.path, point_cloud)

        self.mask_dir = "%s/%s" % (self.path, "mask_indices.npy")
        # if os.path.exists(self.mask_dir):
        #     print("mask operation")
        #     self.mask_indices = np.load(self.mask_dir)[:, None].astype(np.float32)
        #     self.mask_indices = torch.tensor(self.mask_indices).cuda()
        # else:
        #     self.mask_indices = None
        self.mask_indices = None
    
    def load_poses(self):
        poses = []
        if "star" in self.orientation_dir:
            df = starfile.read(self.orientation_dir)
            particle_df = df['particles']
            for i in range(self.particle_num):
                iparticle = particle_df.iloc[i]
                ipose_rot = relion_angle_to_matrix([iparticle.rlnAngleRot, iparticle.rlnAngleTilt, iparticle.rlnAnglePsi])
                if "rlnOriginX" in particle_df.columns and "rlnOriginY" in particle_df.columns:
                    ipose_tran = np.array([iparticle.rlnOriginX, iparticle.rlnOriginY]) / self.size_scale
                elif "rlnOriginXAngst" in particle_df.columns and "rlnOriginYAngst" in particle_df.columns:
                    ipose_tran = np.array([iparticle.rlnOriginXAngst, iparticle.rlnOriginYAngst]) / self.pixel_size / self.size_scale
                poses.append([ipose_rot, ipose_tran])
        elif "npy" in self.orientation_dir:
            df = np.load(self.orientation_dir).astype(np.float64)
            if df.shape[1] == 5:
                angles = df[:, :3]
                trans = df[:, 3:] / self.size_scale
                rots = R_from_relion(angles)
            elif df.shape[1] == 11:
                rots = df[:, :9].reshape(-1, 3, 3)
                trans = df[:, 9:] / self.size_scale
            else:
                print("Error: the dimension of pose")
            for i in range(self.particle_num):
                    poses.append([rots[i], trans[i]])
        self.orientations = poses

    def init_cfg(self):
        with open(self.cfg_file, "r") as f:
            meta_data = json.load(f)

        self.scanner_cfg = meta_data['cfg']
        if not "dVoxel" in self.scanner_cfg:
            self.scanner_cfg["dVoxel"] = list(
                np.array(self.scanner_cfg["sVoxel"])
                / np.array(self.scanner_cfg["nVoxel"])
            )
        self.bbox = torch.tensor(meta_data['bbox'])
        self.pixel_size = self.scanner_cfg['pixelSize']


    def init_Gaussians(self, scale_bound=None, max_density=None):
        out = np.load(self.point_dir)
        self.gaussians = GaussianModel(scale_bound=scale_bound, max_density=max_density)
        out[:, :3] /= self.size_scale * self.pixel_size
        new_points = out.copy()
        new_points[:, 0] = out[:, 2]  ### Note: z y x --> x y z
        new_points[:, 2] = out[:, 0]

        self.gaussians.create_from_pcd(new_points[:, :3], new_points[:, 3:4], 1.0)

    def load_Gaussians(self, scale_bound=None, max_density=None):
        self.gaussians = GaussianModel(scale_bound=scale_bound, max_density=max_density)
        self.gaussians.load_ply(self.point_dir)

    def save(self, epoch, queryfunc, output_path=None):
        if output_path is None:
            point_cloud_path = osp.join(
                self.path, f"point_cloud_{self.half_name}", f"epoch_{epoch}"
            )
        else:
            point_cloud_path = osp.join(
                output_path, f"point_cloud_{self.half_name}", f"epoch_{epoch}"
            )
        self.gaussians.save_ply(osp.join(point_cloud_path, "point_cloud.ply"))
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            # np.save(osp.join(point_cloud_path, "vol_pred.npy"), t2a(vol_pred))
            vol_pred_mrc = osp.join(point_cloud_path, "vol_pred.mrc")
            with mrcfile.new(vol_pred_mrc, overwrite=True) as mrc:
                mrc.set_data(t2a(vol_pred).astype(np.float32))
                mrc.voxel_size = self.pixel_size


def real_space_cryonerf(output, gt, ctf, mask=None, ctf_sign=False):
    D = output.shape[-1]
    output = torch.fft.rfft2(torch.fft.fftshift(output, dim=(-2, -1)), dim=(-2, -1))
    ctf = torch.fft.fftshift(ctf, dim=(-2, -1))
    half_ctf = ctf[..., :D//2+1]
    output = torch.multiply(output, half_ctf)
    output = torch.fft.ifftshift(torch.fft.irfft2(output, dim=(-2, -1)), dim=(-2, -1))
    if ctf_sign:
        gt = torch.fft.rfft2(torch.fft.fftshift(gt, dim=(-2, -1)), dim=(-2, -1))
        gt = torch.multiply(gt, torch.sign(half_ctf))
        gt = torch.fft.ifftshift(torch.fft.irfft2(gt, dim=(-2, -1)), dim=(-2, -1))
    if mask is not None:
        mask = mask.view(1, D, D)
        return F.l1_loss(output[mask == 1], gt[mask == 1])
        # return F.mse_loss(output[mask == 1], gt[mask == 1])
    return F.l1_loss(output, gt)
    # return F.mse_loss(output, gt)


def save_checkpoint(model, epoch, outdir, z=None):
    """Save model weights, latent encoding z, and decoder volumes"""
    mkdir_p(os.path.dirname(outdir))
    # save model weights
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        },
        outdir+'/weights.{}.pkl'.format(epoch),
    )
    # save z
    if z is not None:
        output_pkl = outdir+'/z.{}.pkl'.format(epoch)
        with open(output_pkl, "wb") as f:
            pickle.dump(z, f)


def geometric_loss_radius(gaussian_embedding, base_positions, displacements, neighbor_indices_small, neighbor_indices, raw_neighbor_small_distance, raw_neighbor_distance, mean_neighbor_dist, radius_weight=None, radius_large_weight=None):

    neighbor_norm_loss = torch.mean(torch.norm(gaussian_embedding[:, neighbor_indices[0]] - gaussian_embedding[:, neighbor_indices[1]], p=2, dim=2) * radius_large_weight)

    deformed_positions = base_positions.unsqueeze(0) + displacements

    neighbor_coherence_differences = torch.norm(deformed_positions[:, neighbor_indices[0]] - deformed_positions[:, neighbor_indices[1]], p=2, dim=2)
    neighbor_coherence_loss = torch.mean((neighbor_coherence_differences - raw_neighbor_distance) ** 2 * radius_large_weight)


    neighbor_repulsion_differences = torch.norm(deformed_positions[:, neighbor_indices_small[0]] - deformed_positions[:, neighbor_indices_small[1]], p=2, dim=2)
    neighbor_repulsion_loss = torch.mean(torch.abs(neighbor_repulsion_differences - mean_neighbor_dist))

    displacement_orientation_loss = torch.mean(F.cosine_similarity(displacements[:, neighbor_indices[0]], displacements[:, neighbor_indices[1]], dim=2) * radius_large_weight) * (-1.)
    return neighbor_norm_loss, neighbor_coherence_loss, displacement_orientation_loss, neighbor_repulsion_loss


def weight_assignment(dist_ratio):
    weight = torch.exp(-dist_ratio * 1.0 + 1.0)
    # weight = torch.exp(-dist_ratio * 0.5 + 0.5)
    return weight


def training_EM_heter(
    dataset: Particles,
    opt: OptimizationParams_EM,
    model,
    checkpoint,
    output_dir,
    batch_size,
    zdim
):
    first_iter = 0
    # seed_torch()
    # Set up some parameters
    scanner_cfg = dataset.scanner_cfg
    bbox = dataset.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None
    if scanner_cfg["scale_max"] and scanner_cfg["scale_min"]:
        scale_bound = np.array([scanner_cfg["scale_min"], scanner_cfg["scale_max"]]) * volume_to_world

    # Set up Gaussians
    dataset.load_Gaussians(scale_bound=scale_bound)
    gaussians = dataset.gaussians
    N = gaussians._density.shape[0]
    B = batch_size
    D = dataset.particle_size
    std = scanner_cfg['std']
    mask_indices = dataset.mask_indices[None, :] if dataset.mask_indices is not None else None

    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if output_dir is not None:
        save_path = osp.join(dataset.path, output_dir)

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(save_path, f"ckpt_{dataset.half_name}")
    os.makedirs(ckpt_save_path, exist_ok=True)
    total_iterations = opt.epoch * dataset.particle_num
    progress_bar = tqdm(range(0, total_iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1

    fourier_mask = dataset.ctf_mask

    data_generator = make_dataloader(
        dataset.images,
        batch_size=B,
        num_workers=1,
        shuffler_size=0,
        shuffle=True
    )

    count = 0
    loss = defaultdict(float)
    loss['total'] = 0.0

    gaussians1 = SimpleGaussian(scale_bound, ['density', 'scaling', 'position'], gaussians.max_density)
    gaussians1.create_from_Gaussian(gaussians)
    gaussians2 = SimpleGaussian(scale_bound, ['density', 'scaling', 'position'], gaussians.max_density)
    gaussians2.create_from_Gaussian(gaussians)

    optim = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    optim.zero_grad()

    base_density = gaussians1._density.clone().detach()
    base_scaling = gaussians1._scaling.clone().detach()
    base_xyz = gaussians1._xyz.clone().detach()
    base_pos = gaussians1.get_xyz.clone().detach()
    count = 0

    latent_bank = torch.zeros((dataset.particle_num, zdim)).cuda()
    sample_num = 128 * 50
    k_sample = 128
    sample_ind = np.random.choice(len(latent_bank), size=sample_num, replace=False)
    sample_latent = latent_bank[sample_ind].clone().detach()
    eps = 1e-4

    points = base_pos.cpu().numpy()
    tree = cKDTree(points)
    knn_dists, knn_indices = tree.query(points, k=2)

    mean_dist = np.mean(knn_dists[:, 1])
    neighbor_radius = mean_dist * 1.5
    # neighbor_radius_large = mean_dist * 3.0
    neighbor_radius_large = mean_dist * 2.0
    
    radius_graph = radius_neighbors_graph(points, radius=neighbor_radius, n_jobs=8)
    radius_graph_coo = radius_graph.tocoo()
    rg_col = torch.tensor(radius_graph_coo.col)
    rg_row = torch.tensor(radius_graph_coo.row)
    # radius_indices = torch.stack([rg_col, rg_col]).long()
    radius_indices = torch.stack([rg_col, rg_row]).long().cuda()
    raw_neighbor_distance = torch.norm(base_pos[radius_indices[0]] - base_pos[radius_indices[1]], p=2, dim=1)

    radius_graph_large = radius_neighbors_graph(points, radius=neighbor_radius_large, n_jobs=8)
    radius_graph_large_coo = radius_graph_large.tocoo()
    rg_large_col = torch.tensor(radius_graph_large_coo.col)
    rg_large_row = torch.tensor(radius_graph_large_coo.row)
    radius_large_indices = torch.stack([rg_large_col, rg_large_row]).long().cuda()

    raw_neighbor_large_distance = torch.norm(base_pos[radius_large_indices[0]] - base_pos[radius_large_indices[1]], p=2, dim=1)
    raw_dist_large_weight = weight_assignment(raw_neighbor_large_distance / mean_dist)

    radius_dist_weight = weight_assignment(raw_neighbor_distance / mean_dist)

    record_loss = defaultdict(float)

    for epoch in range(opt.epoch):

        model.train()
        for iteration, minibatch in enumerate(data_generator):
            iter_start.record()
            count += 1
            indices = minibatch[-1]
            original_image = minibatch[0]

            # # Compute loss
            gt_image = original_image.cuda()
            gt_image = gt_image / std
            inputted_img = gt_image
            gaussian_param, z_mu, z_logvar = model(inputted_img, base_density, base_scaling, base_xyz)

            latent_bank[indices] = z_mu.detach()
            if epoch > 1:
                sample_latent = latent_bank[sample_ind].clone().detach()

                diff = (z_mu.unsqueeze(1) - sample_latent.unsqueeze(0)).pow(2).sum(-1)
                top = torch.topk(diff, k=k_sample, largest=False, sorted=True, dim=1)
                top_latent = sample_latent[top.indices]
                neg = torch.topk(diff, k=k_sample*5, largest=True, sorted=True, dim=1)
                uni_sample = np.random.choice(len(sample_latent), size=k_sample*5, replace=False)
                uni_latent = sample_latent[uni_sample].unsqueeze(0)
                neg_latent = sample_latent[neg.indices]
                neg_latent = 0.8*neg_latent + 0.2*uni_latent
                top_diff = (z_mu.unsqueeze(1) - top_latent).pow(2) + eps
                neg_diff = (z_mu.unsqueeze(1) - neg_latent).pow(2) + eps
                knn_top_loss = torch.log(1 + top_diff).mean()
                knn_neg_loss = torch.log(1 + 1./neg_diff).mean()
                knn_loss = knn_top_loss + knn_neg_loss * 5

                knn_loss /= 1e6
                loss["knn"] += knn_loss

                sample_ind = np.random.choice(len(latent_bank), size=sample_num, replace=False)
            else:
                knn_loss = torch.tensor(0.).cuda()

            gaussian_param_xyz = gaussian_param[2]
            if mask_indices is not None:
                gaussian_param_xyz *= mask_indices
            gaussian_param_density = gaussian_param[0]
            gaussian_param_scaling = gaussian_param[1]
            gaussian_embedding = gaussian_param[-1]

            render_img_list = []
            ctf_list = []
            for i in range(len(indices)):
                ind = indices[i]
                orientation = dataset.orientations[ind]
                ctf_setting = ctf_params(dataset.ctf_param[ind])
                ctf = compute_single_ctf(dataset.particle_size, ctf_setting).float()
                ctf *= fourier_mask
                ctf_list.append(ctf)
                
                gaussians2._density = base_density + gaussian_param_density[i]
                gaussians2._scaling = base_scaling
                gaussians2._xyz = base_xyz + gaussian_param_xyz[i]

                updated_render_pkg = render(orientation, gaussians2, dataset.particle_size)
                updated_image, updated_viewspace_point_tensor, updated_visibility_filter, updated_radii = (
                    updated_render_pkg["render"],
                    updated_render_pkg["viewspace_points"],
                    updated_render_pkg["visibility_filter"],
                    updated_render_pkg["radii"],
                )
                render_img_list.append(updated_image)

            render_imgs = torch.cat(render_img_list, dim=0).cuda()
            ctfs = torch.stack(ctf_list, dim=0).cuda()
            render_loss = real_space_cryonerf(render_imgs, gt_image.detach(), ctfs.detach(), ctf_sign=False)

            loss["render"] += render_loss
            loss["total"] = loss["total"] + render_loss

            kld_loss = torch.mean(
                -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0
            )

            kld_loss /= 5e4
            loss["kld"] += kld_loss
            loss["total"] = loss["total"] + kld_loss

            neighbor_norm_loss, neighbor_coherence_loss, displacement_orientation_loss, neighbor_repulsion_loss = geometric_loss_radius(gaussian_embedding, base_pos, gaussian_param_xyz, radius_indices, radius_large_indices, raw_neighbor_distance, raw_neighbor_large_distance, mean_dist, radius_weight=radius_dist_weight, radius_large_weight=raw_dist_large_weight)

            neighbor_norm_loss /= 1e5
            neighbor_coherence_loss /= 1e2
            displacement_orientation_loss /= 1e5
            neighbor_repulsion_loss /= 1e3

            loss["neighbor_norm"] += neighbor_norm_loss
            loss["neighbor_coherence_loss"] += neighbor_coherence_loss
            loss["displacement_orient_loss"] += displacement_orientation_loss
            loss["repulsion_loss"] += neighbor_repulsion_loss

            deformation_loss = neighbor_norm_loss + neighbor_coherence_loss + displacement_orientation_loss + neighbor_repulsion_loss

            if knn_loss is not None:
                total_loss = render_loss + kld_loss + knn_loss + deformation_loss
            else:
                total_loss = render_loss + kld_loss + deformation_loss

            optim.zero_grad()

            total_loss.backward()

            optim.step()

            for k in loss.keys():
                record_loss[k] = loss[k].item()
                loss[k] = 0.0

            iter_end.record()

        for k in record_loss.keys():
            print(k, record_loss[k])

        with torch.no_grad():
            model.eval()
            z = None
            outdir = osp.join(dataset.path, output_dir)
            save_checkpoint(model, epoch, outdir, z)


def eval_model_output(
    dataset: Particles,
    model,
    output_dir,
    model_weight,
    eval_z=None
):
    first_iter = 0
    # Set up some parameters
    scanner_cfg = dataset.scanner_cfg
    bbox = dataset.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])

    scale_bound = None
    if scanner_cfg["scale_max"] and scanner_cfg["scale_min"]:
        scale_bound = np.array([scanner_cfg["scale_min"], scanner_cfg["scale_max"]]) * volume_to_world

    # Set up Gaussians
    dataset.load_Gaussians(scale_bound)
    gaussians = dataset.gaussians
    N = gaussians._density.shape[0]
    D = dataset.particle_size
    std = scanner_cfg['std']
    mask_indices = dataset.mask_indices if dataset.mask_indices is not None else None

    fourier_mask = dataset.ctf_mask

    data_generator = make_dataloader(
        dataset.images,
        batch_size=1,
        num_workers=0,
        shuffler_size=0,
    )

    model_file = osp.join(output_dir, model_weight)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["model_state_dict"])

    gaussians1 = SimpleGaussian(scale_bound, ['density', 'scaling', 'position'], gaussians.max_density)
    gaussians1.create_from_Gaussian(gaussians)
    gaussians2 = SimpleGaussian(scale_bound, ['density', 'scaling', 'position'], gaussians.max_density)
    gaussians2.create_from_Gaussian(gaussians)
    del gaussians

    model.eval()
    if eval_z is None:

        progress_bar = tqdm(range(0, dataset.particle_num), desc="eval", leave=False)
        progress_bar.update(first_iter)
        first_iter += 1

        z_mu_all = []
        ind_all = []
        gaussian_variation = []
        gaussian_embedding = None

        base_density = gaussians1._density.clone().detach()
        base_scaling = gaussians1._scaling.clone().detach()
        base_xyz = gaussians1._xyz.clone().detach()
        base_pos = gaussians1.get_xyz.clone().detach()
        base_info = torch.cat([base_density, base_scaling, base_pos], dim=-1)

        for iteration, minibatch in enumerate(data_generator):
            ind = minibatch[-1]
            original_image = minibatch[0].cuda()
            ctf_setting = ctf_params(dataset.ctf_param[ind])
            ctf = compute_single_ctf(dataset.particle_size, ctf_setting).float()
            ctf *= fourier_mask
            gt_image = original_image.cuda()
            # gt_image = gt_image / 8.7975
            gt_image = gt_image / std
            inputted_img = gt_image

            gaussian_param, z_mu, z_logvar = model(inputted_img, base_density, base_scaling, base_xyz)

            gaussian_param_xyz = gaussian_param[2].view(N, 3)
            gaussian_param_density = gaussian_param[0].view(N, 1)
            save_gaussian_variation = torch.cat([gaussian_param_density, gaussian_param_xyz], dim=1)
            save_gaussian_variation = save_gaussian_variation.detach().cpu().numpy()
            gaussian_variation.append(save_gaussian_variation)

            gaussian_embedding = gaussian_param[-1]

            z_mu_all.append(z_mu.detach().cpu().numpy())
            ind_all.append(ind)

            progress_bar.update(1)

            del gt_image
            del gaussian_param
            del z_mu

        z_mu_all = np.vstack(z_mu_all)
        ind_all = np.vstack(ind_all)

        gaussian_variation = np.stack(gaussian_variation, axis=0)
        gaussian_embedding = gaussian_embedding.detach().cpu().numpy()

        g_variation_path = osp.join(output_dir, "%s.output.gaussian_variation.npy" % model_weight[:-4])
        g_embedding_path = osp.join(output_dir, "%s.output.gaussian_embedding.npy" % model_weight[:-4])
        np.save(g_variation_path, gaussian_variation)
        np.save(g_embedding_path, gaussian_embedding)

        save_path = osp.join(output_dir, "%s.output.%s" % (model_weight[:-4], model_weight[-3:]))
        output = {"z_mu": z_mu_all, "ind": ind_all}
        with open(save_path, "wb") as f:
            pickle.dump(output, f)
    else:

        base_density = gaussians1._density.clone().detach()
        activate_density = gaussians1.get_density.clone().detach()
        base_scaling = gaussians1._scaling.clone().detach()
        activate_scaling = gaussians1.get_scaling.clone().detach()
        base_xyz = gaussians1._xyz.clone().detach()
        base_pos = gaussians1.get_xyz.clone().detach()
        eval_z = torch.tensor(eval_z, device="cuda").view(1, -1)

        base_info = torch.cat([base_density, base_scaling, base_xyz], dim=-1)
        gaussian_param = model.decode(eval_z, base_info)

        gaussian_param_xyz = gaussian_param[2].view(N, 3)
        if mask_indices is not None:
            gaussian_param_xyz *= mask_indices
        gaussian_param_density = gaussian_param[0].view(N, 1)
        gaussian_param_scaling = gaussian_param[1].view(N, 1)

        gaussians2._density = base_density + gaussian_param_density
        gaussians2._scaling = base_scaling
        gaussians2._xyz = base_xyz + gaussian_param_xyz
            
        queryfunc = lambda x: query(
            x,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            reorder=True
        )

        vol_pred = queryfunc(gaussians2)["vol"]
        if eval_z.shape[1] == 1:
            vol_pred_mrc = osp.join(output_dir, "%s.output.z_%.4f.mrc" % (model_weight[:-4], eval_z.cpu().data))
        else:
            vol_pred_mrc = osp.join(output_dir, "%s.output.z_%.4f.mrc" % (model_weight[:-4], eval_z.cpu().data[0, 0]))
            
        with mrcfile.new(vol_pred_mrc, overwrite=True) as mrc:
            mrc.set_data(t2a(vol_pred).astype(np.float32))
            mrc.voxel_size = dataset.pixel_size



if __name__ == "__main__":

    #### train 3DGS halfmap

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams_EM(parser)
    op = OptimizationParams_EM(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--half_name1", type=str, default=None)
    parser.add_argument("--half_name2", type=str, default=None)
    parser.add_argument("--pose", type=str, default=None)
    parser.add_argument("--ctf", type=str, default=None)
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--zdim", type=int, default=8)
    parser.add_argument("--no_window", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    args.save_epochs.append(args.epoch)
    safe_state(args.quiet)
    # print(args)

    args.lr = 1e-4
    args.wd = 0
    args.epoch = 20

    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print("Use total data")
    window_flag = not args.no_window
    dataset = Particles(args.source_path, args.particle_name, args.half_name1, args.pose, args.ctf, args.point, args.cfg, window=window_flag)

    pos_enc_dim = 0
    D = dataset.particle_size
    gaussian_embedding_dim = 32
    model = HetGMMVAESimpleIndependent(
        qlayers=3,
        qdim=1024,
        in_dim=D * D,
        zdim=args.zdim,
        n_clusters=args.nclass,
        gaussian_embedding_dim=gaussian_embedding_dim,
        gaussian_kdim=64,
        gaussian_klayers=3,
        feature_dim=256,
        feature_kdim=64,
        feature_klayers=2,
        activation=nn.ReLU,
        archi_type='MLP'
    )
    model.to(device="cuda")

    if not args.check:

        training_EM_heter(
            dataset,
            op.extract(args),
            model,
            args.start_checkpoint,
            args.output,
            args.batch_size,
            args.zdim,
        )
        print("Gaussians trained complete.")


    #### test 3DGS

    else:

        output_dir = osp.join(args.source_path, args.output)
        model_epoch = args.epoch - 1
        model_weight = "weights.%d.pkl" % model_epoch
        eval_model_output(dataset, model, output_dir, model_weight)

        output_file = osp.join(output_dir, "weights.%d.output.pkl" % model_epoch)
        with open(output_file, 'rb') as f:
            model_output = pickle.load(f)
        # print(model_output)
        z_mu = model_output["z_mu"]

        K = 10
        kmeans_labels, centers = cluster_kmeans(z_mu, K)
        centers, centers_ind = get_nearest_point(z_mu, centers)

        print(centers)
        print(centers_ind)
        center_dir = "%s/output_z_%d" % (output_dir, model_epoch)
        print(centers.shape)
        np.save(center_dir, centers)

        for z_value in centers:
            # print(z_value.dtype)
            eval_model_output(dataset, model, output_dir, model_weight, eval_z=z_value)



        # output_dir = osp.join(args.source_path, args.output)
        # model_epoch = args.epoch - 1
        # model_weight = "weights.%d.pkl" % model_epoch

        # output_file = osp.join(output_dir, "weights.%d.output.pkl" % model_epoch)
        # with open(output_file, 'rb') as f:
        #     model_output = pickle.load(f)
        # # print(model_output)
        # z_mu = model_output["z_mu"]

        # from eval_heter import run_pca, get_pc_traj
        # pc, pca = run_pca(z_mu)
        # dim = 1
        # lim = [10, 90]
        # pc_values = np.percentile(pc[:, dim - 1], np.linspace(lim[0], lim[1], 10, endpoint=True))
        # traj = get_pc_traj(pca, args.zdim, 10, dim, None, None, pc_values)

        # for z_value in traj:
        #     # print(z_value.dtype)
        #     z_value = z_value.astype(np.float32)
        #     eval_model_output(dataset, model, output_dir, model_weight, eval_z=z_value)
