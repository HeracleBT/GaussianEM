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

sys.path.append("./")
from arguments import ModelParams_EM, OptimizationParams_EM, PipelineParams
from utils.general_utils import safe_state
from utils.general_utils import load_config
from utils.general_utils import t2a

from model.gaussian_model_insight import GaussianModel
from model.render_query_EM import query, render
from utils.utils_em import circular_mask, R_from_relion, relion_angle_to_matrix
from data.dataset_em import ImageDataset, make_dataloader
from particle_preprocess import compute_single_ctf, ctf_params

epsilon = 0.0000005

class Particles:
    gaussians: GaussianModel

    def __init__(self, dataset_dir, particle_name, pose_file, ctf_file, point_cloud, cfg, window=True):
        self.path = dataset_dir
        self.particle_name = particle_name
        self.images_dir = "%s/%s.mrcs" % (self.path, particle_name)
        self.orientation_dir = "%s/%s" % (self.path, pose_file)
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
        self.gaussians = GaussianModel(scale_bound, max_density=max_density)
        out[:, :3] /= self.size_scale * self.pixel_size
        new_points = out.copy()
        new_points[:, 0] = out[:, 2]  ### Note: z y x --> x y z
        new_points[:, 2] = out[:, 0]

        # self.gaussians.create_from_pcd(new_points[:, :3], new_points[:, 3:4], 1.0)

        densities = new_points[:, 3:4] * 5
        densities = np.clip(densities, None, 15.0)
        self.gaussians.create_from_pcd(new_points[:, :3], densities, 1.0)   ### fit image


    def save(self, epoch, queryfunc, output_path=None):
        if output_path is None:
            point_cloud_path = osp.join(
                self.path, f"point_cloud_{self.particle_name}", f"epoch_{epoch}"
            )
        else:
            point_cloud_path = osp.join(
                output_path, f"point_cloud_{self.particle_name}", f"epoch_{epoch}"
            )
        self.gaussians.save_ply(osp.join(point_cloud_path, "point_cloud.ply"))
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_pred_mrc = osp.join(point_cloud_path, "vol_pred.mrc")
            with mrcfile.new(vol_pred_mrc, overwrite=True) as mrc:
                mrc.set_data(t2a(vol_pred).astype(np.float32))
                mrc.voxel_size = self.pixel_size

    def save_prune(self, epoch, queryfunc, output_path=None):
        if output_path is None:
            point_cloud_path = osp.join(
                self.path, f"point_cloud_{self.particle_name}", f"epoch_{epoch}"
            )
        else:
            point_cloud_path = osp.join(
                output_path, f"point_cloud_{self.particle_name}", f"epoch_{epoch}"
            )
        self.gaussians.save_ply(osp.join(point_cloud_path, "point_cloud_after_prune.ply"))
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_pred_mrc = osp.join(point_cloud_path, "vol_pred_after_prune.mrc")
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


def prune_low_contribution_gaussians(gaussians, orientations, particle_size, K=5, prune_ratio=0.1):
    top_list = [None, ] * K
    for i, pose in enumerate(orientations):
        trans_pkg = render(pose, gaussians, particle_size, record_transmittance=True)
        trans = trans_pkg.detach()
        if top_list[0] is not None:
            m = trans > top_list[0]
            if m.any():
                for i in range(K - 1):
                    top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
                top_list[0][m] = trans[m]
        else:
            top_list = [trans.clone() for _ in range(K)]

        del trans_pkg

    contribution = torch.stack(top_list, dim=-1).mean(-1)
    tile = torch.quantile(contribution, prune_ratio)
    prune_mask = contribution < tile
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()
    return


def training_EM_homo(
    dataset: Particles,
    scene: ModelParams_EM,
    opt: OptimizationParams_EM,
    pipe: PipelineParams,
    checkpoint,
    output_dir,
    batch_size
):
    first_iter = 0
    # Set up some parameters
    scanner_cfg = dataset.scanner_cfg
    bbox = dataset.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if scanner_cfg["scale_max"] and scanner_cfg["scale_min"]:
        scale_bound = np.array([scanner_cfg["scale_min"], scanner_cfg["scale_max"]]) * volume_to_world

    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        reorder=True
    )

    # Set up Gaussians
    dataset.init_Gaussians(scale_bound=scale_bound)
    dataset.gaussians.training_setup(opt)
    gaussians = dataset.gaussians
    std = scanner_cfg['std']

    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if output_dir is not None:
        save_path = osp.join(dataset.path, output_dir)


    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(save_path, f"ckpt_{dataset.particle_name}")
    os.makedirs(ckpt_save_path, exist_ok=True)
    total_iterations = opt.epoch * dataset.particle_num
    progress_bar = tqdm(range(0, total_iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    fourier_mask = dataset.ctf_mask

    data_generator = make_dataloader(
        dataset.images,
        batch_size=1,
        num_workers=1,
        shuffler_size=0,
        shuffle=True,
    )

    batch_num = batch_size
    count = 0
    loss = defaultdict(float)
    loss['total'] = 0.0

    for epoch in range(opt.epoch):

        for iteration, minibatch in enumerate(data_generator):
            iter_start.record()
            count += 1
            # Update learning rate
            true_ite = iteration + epoch * dataset.particle_num + 1
            gaussians.update_learning_rate(true_ite)
            ind = minibatch[-1]
            original_image = minibatch[0].cuda()
            orientation = dataset.orientations[ind]
            # print(dataset.ctf_param[ind].shape)
            ctf_setting = ctf_params(dataset.ctf_param[ind])
            ctf = compute_single_ctf(dataset.particle_size, ctf_setting).float()
            ctf *= fourier_mask

            # Render projection
            render_pkg = render(orientation, gaussians, dataset.particle_size)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            # Compute loss
            gt_image = original_image.cuda()
            gt_image = gt_image / std
            render_loss = real_space_cryonerf(image, gt_image.detach(), ctf.detach(), ctf_sign=False)

            total_loss = render_loss / batch_num
            total_loss.backward()

            loss["render"] += render_loss.item()

            if count % batch_num == 0:
                count = 0
                loss['total'] = loss['render']

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'] / batch_num:.1e}",
                        "render": f"{loss['render'] / batch_num:.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(batch_num)

                for k in loss.keys():
                    loss[k] = 0.0
            else:
                continue

            iter_end.record()

            with torch.no_grad():
                # Adaptive control
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if true_ite < opt.densify_until_iter:
                    if (
                        true_ite > opt.densify_from_iter
                        and true_ite % opt.densification_interval == 0
                    ):  
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.density_min_threshold,
                            opt.max_screen_size,
                            max_scale,
                            opt.max_num_gaussians,
                            densify_scale_threshold,
                            bbox,
                        )

                if gaussians.get_density.shape[0] == 0:
                    raise ValueError(
                        "No Gaussian left. Change adaptive control hyperparameters!"
                    )

                # Optimization
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # Logging
                metrics = {}
                for l in loss:
                    if isinstance(loss[l], float):
                        metrics["loss_" + l] = loss[l]
                    else:
                        metrics["loss_" + l] = loss[l].item()
                for param_group in gaussians.optimizer.param_groups:
                    metrics[f"lr_{param_group['name']}"] = param_group["lr"]

        tqdm.write(f"[ITER {epoch}] Saving Gaussians, perform {true_ite} iterations")
        dataset.save(epoch, queryfunc, save_path)

        # # if epoch < opt.epoch // 2:
        # if epoch < 6:

        #     total_sample_indices = [i for i in range(dataset.particle_num)]
        #     random.shuffle(total_sample_indices)
        #     sample_indices = total_sample_indices[:1000]
        #     orientation_subset = []
        #     for s in sample_indices:
        #         orientation_subset.append(dataset.orientations[s])

        #     prune_low_contribution_gaussians(gaussians, orientation_subset, dataset.particle_size, K=5, prune_ratio=opt.contribution_prune_ratio)
        #     print(f'Num gs after contribution prune: {len(gaussians.get_xyz)}')
        #     dataset.save_prune(epoch, queryfunc, save_path)


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


    ### IgG-1D insight scale [0.5, 1.0] settings
    if "IgG-1D" in args.source_path:
        print("IgG-1D dataset")
        args.position_lr_max_steps = 1_000_000
        args.density_lr_max_steps =  1_000_000
        args.scaling_lr_max_steps =  1_000_000
        args.rotation_lr_max_steps =  1_000_000
        args.densify_until_iter =  1_000_000
        args.densify_grad_threshold = 5.0e-5
        args.densify_scale_threshold = 0.0046
        args.max_scale = 0.005


    ### 10180 scale [0.5, 1.0] settings 256
    if "10180" in args.source_path:
        print("EMPAIR 10180 dataset")
        args.position_lr_max_steps = 3_270_000
        args.density_lr_max_steps =  3_270_000
        args.scaling_lr_max_steps =  3_270_000
        args.rotation_lr_max_steps =  3_270_000
        args.densify_until_iter =  3_270_000
        args.densify_grad_threshold = 5.0e-5
        args.densify_scale_threshold = 0.0078
        args.max_scale = 0.008
        args.contribution_prune_ratio = 0.1
        # args.max_num_gaussians = 10000


    ### 10841 scale [0.25, 0.75] settings
    if "10841" in args.source_path:
        print("EMPAIR 10841 dataset")
        args.position_lr_max_steps = 1_238_000
        args.density_lr_max_steps =  1_238_000
        args.scaling_lr_max_steps =  1_238_000
        args.rotation_lr_max_steps =  1_238_000
        args.densify_until_iter =  1_238_000
        args.densify_grad_threshold = 5.0e-5
        args.densify_scale_threshold = 0.006
        args.max_scale = 0.0063
        args.contribution_prune_ratio = 0.1


    ### 10059 scale [0.5, 1.0] settings
    if "10059" in args.source_path:
        print("10059 dataset")
        args.position_lr_max_steps = 2_187_870
        args.density_lr_max_steps =  2_187_870
        args.scaling_lr_max_steps =  2_187_870
        args.rotation_lr_max_steps =  2_187_870
        args.densify_until_iter =  2_187_870
        args.densify_grad_threshold = 5.0e-5
        args.densify_scale_threshold = 0.0043
        args.max_scale = 0.0065
        args.contribution_prune_ratio = 0.1


    ### 10345 scale [0.5, 1.0] settings
    if "10345" in args.source_path:
        print("10345 dataset")
        args.position_lr_max_steps = 842_660
        args.density_lr_max_steps =  842_660
        args.scaling_lr_max_steps =  842_660
        args.rotation_lr_max_steps =  842_660
        args.densify_until_iter =  842_660
        args.densify_grad_threshold = 5.0e-5
        args.densify_scale_threshold = 0.0048
        args.max_scale = 0.005
        args.contribution_prune_ratio = 0.3


    ### gaoxiang_group scale [0.5, 1.0] settings
    if "gaoxiang_group" in args.source_path:
        print("gaoxiang_group dataset")
        args.position_lr_max_steps = 1_065_640
        args.density_lr_max_steps =  1_065_640
        args.scaling_lr_max_steps =  1_065_640
        args.rotation_lr_max_steps =  1_065_640
        args.densify_until_iter =  1_065_640
        args.densify_grad_threshold = 5.0e-5
        args.densify_scale_threshold = 0.0019
        args.max_scale = 0.002
        args.contribution_prune_ratio = 0.3


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
    training_EM_homo(
        dataset,
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.start_checkpoint,
        args.output,
        args.batch_size
    )
    print("Gaussians trained complete.")
