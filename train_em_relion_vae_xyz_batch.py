import os
import os.path as osp
import torch
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import mrcfile
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
from data.dataset_em import make_dataloader
from particle_preprocess import compute_single_ctf, ctf_params
from model.gmmvae_heter import HetGMMVAESimpleIndependent
from torch import nn
from utils.general_utils import mkdir_p
from scipy.spatial import cKDTree
from sklearn.neighbors import radius_neighbors_graph
from data.training_particles import Particles

epsilon = 0.0000005

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
    checkpoint,
    output_dir,
    batch_size
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
    zdim = scanner_cfg['zdim']

    gaussian_embedding_dim = scanner_cfg['gaussian_embedding_dim']
    model = HetGMMVAESimpleIndependent(
        qlayers=scanner_cfg['qlayers'],
        qdim=scanner_cfg['qdim'],
        in_dim=D * D,
        zdim=zdim,
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

    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if output_dir is not None:
        save_path = osp.join(dataset.path, output_dir)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
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

            progress_bar.set_postfix(
                {
                    # "render": f"{loss['render'].item():.1e}",
                    "kld": f"{loss['kld'].item():.1e}",
                    "nbr_nm": f"{loss['neighbor_norm'].item():.1e}",
                    "nbr_co": f"{loss['neighbor_coherence_loss'].item():.1e}",
                    "dplace": f"{loss['displacement_orient_loss'].item():.1e}",
                    "repul": f"{loss['repulsion_loss'].item():.1e}",
                }
            )
            progress_bar.update(B)

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

    return model


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
        base_scaling = gaussians1._scaling.clone().detach()
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
    parser.add_argument("--particle_name", type=str, default=None)
    parser.add_argument("--pose", type=str, default=None)
    parser.add_argument("--ctf", type=str, default=None)
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_window", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    args.save_epochs.append(args.epoch)
    safe_state(args.quiet)

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
    dataset = Particles(args.source_path, args.particle_name, args.pose, args.ctf, args.point, args.cfg, window=window_flag)

    model = training_EM_heter(
        dataset,
        op.extract(args),
        args.start_checkpoint,
        args.output,
        args.batch_size
    )

    print("Gaussians trained complete.")

    output_dir = osp.join(args.source_path, args.output)
    model_epoch = args.epoch - 1
    model_weight = "weights.%d.pkl" % model_epoch
    eval_model_output(dataset, model, output_dir, model_weight)

    print("Save latent variable.")
