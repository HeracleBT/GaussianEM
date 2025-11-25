import numpy as np
import torch
import mrcfile
import starfile
import os.path as osp
from typing import Optional
from scipy.interpolate import RegularGridInterpolator
from collections.abc import Iterable
from data.relion import Relionfile, ImageSource
import json
from collections import defaultdict
from utils.utils_em import R_from_relion
from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils_em import ht2_center, iht2_center
import os, sys
from data.mrcfile_em import MRCHeader

sys.path.append("./")
from arguments import ModelParams_EM, OptimizationParams_EM, PipelineParams
from utils.general_utils import safe_state
from utils.general_utils import t2a

from model.gaussian_model_insight import GaussianModel
from model.render_query_EM import query, render
from utils.utils_em import circular_mask, R_from_relion, relion_angle_to_matrix
from data.dataset_em import ImageDataset, make_dataloader

epsilon = 0.0000005

class Particles:
    gaussians: GaussianModel

    def __init__(self, dataset_dir, particle_name, pose_file, ctf_file, point_cloud, cfg, window=True):
        self.path = dataset_dir
        self.particle_name = particle_name
        self.images_dir = "%s/%s.mrcs" % (self.path, particle_name)
        # self.orientation_dir = "%s/%s" % (self.path, pose_file)
        self.orientation_dir = pose_file
        # self.cfg_file = "%s/%s" % (self.path, cfg)
        self.cfg_file = cfg
        
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

        # self.ctf_dir = "%s/%s" % (self.path, ctf_file)
        self.ctf_dir = ctf_file
        self.ctf_param = np.load(self.ctf_dir)
        self.ctf_mask = circular_mask(self.particle_size).cuda()
        self.load_poses()
        # self.point_dir = "%s/%s" % (self.path, point_cloud)
        self.point_dir = point_cloud
    
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

class ctf_params:

    snr1: float
    snr2: float
    seed: int
    Apix: float  # Pixel size  (A/pix)
    kv: float  # Microscope voltage  (kv)
    dfu: float  # defocus U  (A)
    dfv: float  # defocus V  (A)
    ang: float  #  Astigmatism angle  (deg)
    cs: float  #  Spherical aberration  (mm)
    wgh: float  #  Amplitude contrast ratio
    ps: float  #  phase shift  (deg)
    b: float  #  b factor for Gaussian envelop  (A^2)

    def __init__(self, parameters: Optional[np.ndarray] = None):
        if parameters is None:
            self.snr1 = 0
            self.snr2 = 0
            self.seed = 0
            self.Apix = 1.0
            self.kv = 300
            self.dfu = 15000
            self.dfv = 15000
            self.ang = 0
            self.cs = 2
            self.wgh = 0.1
            self.ps = 0
            self.b = 100
        else:
            assert len(parameters) == 9
            self.snr1 = 0
            self.snr2 = 0
            self.seed = 0
            self.Apix = parameters[1]
            self.dfu = parameters[2]
            self.dfv = parameters[3]
            self.ang = parameters[4]
            self.kv = parameters[5]
            self.cs = parameters[6]
            self.wgh = parameters[7]
            if np.isnan(parameters[8]):
                self.ps = 0.0
            else:
                self.ps = parameters[8]
            self.b = None

def compute_single_ctf(D, ctf_params):
    freqs = np.arange(-D / 2, D / 2) / (ctf_params.Apix * D)
    x0, x1 = np.meshgrid(freqs, freqs)
    freqs = torch.tensor(np.stack([x0.ravel(), x1.ravel()], axis=1)).cuda()
    ctf = compute_ctf(
        freqs,
        torch.Tensor([ctf_params.dfu]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.dfv]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.ang]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.kv]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.cs]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.wgh]).reshape(-1, 1).cuda(),
        phase_shift = torch.Tensor([ctf_params.ps]).reshape(-1, 1).cuda(),
        bfactor = torch.Tensor([ctf_params.b]).reshape(-1, 1).cuda() if ctf_params.b is not None else None,
    )
    ctf = ctf.reshape((D, D))
    return ctf

def compute_ctf(
    freqs: torch.Tensor,
    dfu: torch.Tensor,
    dfv: torch.Tensor,
    dfang: torch.Tensor,
    volt: torch.Tensor,
    cs: torch.Tensor,
    w: torch.Tensor,
    phase_shift: Optional[torch.Tensor] = None,
    scalefactor: Optional[torch.Tensor] = None,
    bfactor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the 2D CTF

    Input:
        freqs: Nx2 array of 2D spatial frequencies
        dfu: DefocusU (Angstrom)
        dfv: DefocusV (Angstrom)
        dfang: DefocusAngle (degrees)
        volt: accelerating voltage (kV)
        cs: spherical aberration (mm)
        w: amplitude contrast ratio
        phase_shift: degrees
        scalefactor : scale factor
        bfactor: envelope fcn B-factor (Angstrom^2)
    """
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / torch.sqrt(volt + 0.97845e-6 * volt**2)
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.arctan2(y, x)
    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = (
        2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2)
        - phase_shift
    )
    ctf = torch.sqrt(1 - w**2) * torch.sin(gamma) - w * torch.cos(gamma)
    if scalefactor is not None:
        ctf *= scalefactor
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)
    return ctf

def compute_single_ctf(D, ctf_params):
    freqs = np.arange(-D / 2, D / 2) / (ctf_params.Apix * D)
    x0, x1 = np.meshgrid(freqs, freqs)
    freqs = torch.tensor(np.stack([x0.ravel(), x1.ravel()], axis=1)).cuda()
    ctf = compute_ctf(
        freqs,
        torch.Tensor([ctf_params.dfu]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.dfv]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.ang]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.kv]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.cs]).reshape(-1, 1).cuda(),
        torch.Tensor([ctf_params.wgh]).reshape(-1, 1).cuda(),
        phase_shift = torch.Tensor([ctf_params.ps]).reshape(-1, 1).cuda(),
        bfactor = torch.Tensor([ctf_params.b]).reshape(-1, 1).cuda() if ctf_params.b is not None else None,
    )
    ctf = ctf.reshape((D, D))
    return ctf


def standardize(images):
    images = (images - np.mean(images, axis=(1, 2)).reshape(-1, 1, 1)) / np.std(images, axis=(1, 2)).reshape(-1, 1, 1)
    return images


def sign_flip(chunk: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    """data * -1, convert bright to dark."""

    return chunk * -1


def parse_pose_star(input, img_d, img_apix):
    relionfile = Relionfile(input)
    apix, resolution = relionfile.apix, relionfile.resolution
    if img_d is not None:
        resolution = np.array([img_d for _ in range(len(relionfile))])
    if img_apix is not None:
        apix = np.array([img_apix for _ in range(len(relionfile))])
    euler = np.zeros((len(relionfile), 3))
    euler[:, 0] = relionfile.df["_rlnAngleRot"]
    euler[:, 1] = relionfile.df["_rlnAngleTilt"]
    euler[:, 2] = relionfile.df["_rlnAnglePsi"]
    # print("Euler angles (Rot, Tilt, Psi)")
    rot = R_from_relion(euler)

    trans = np.zeros((len(relionfile), 2))
    if "_rlnOriginX" in relionfile.df.columns and "_rlnOriginY" in relionfile.df.columns:
        # translations in pixels
        trans[:, 0] = relionfile.df["_rlnOriginX"]
        trans[:, 1] = relionfile.df["_rlnOriginY"]
    elif (
        "_rlnOriginXAngst" in relionfile.df.columns
        and "_rlnOriginYAngst" in relionfile.df.columns
    ):
        # translation in Angstroms (Relion 3.1)
        if apix is None:
            raise ValueError(
                f"Must provide --Apix argument to convert _rlnOriginXAngst and "
                f"_rlnOriginYAngst translation units as A/px not "
                f"found in `{input}`!"
            )
        trans[:, 0] = relionfile.df["_rlnOriginXAngst"]
        trans[:, 1] = relionfile.df["_rlnOriginYAngst"]
        trans /= apix.reshape(-1, 1)
    else:
        print(
            "Warning: Neither _rlnOriginX/Y nor _rlnOriginX/YAngst found. "
            "Defaulting to 0s."
        )
    return rot, trans


def parse_ctf_star(input, img_d, img_apix):
    stardata = Relionfile(input)
    ctf_params = np.zeros((len(stardata), 9))
    ctf_params[:, 0] = img_d
    ctf_params[:, 1] = img_apix
    for i, header in enumerate(
        [
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnDefocusAngle",
            "_rlnVoltage",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            "_rlnPhaseShift",
        ]
    ):
        ctf_params[:, i + 2] = (
            stardata.get_optics_values(header)
        )
    return ctf_params


def grid_sample_with_value_threshold(data, interval, value_threshold, num_points=None):
    """
    Parameters:
        data (numpy.ndarray): 3D array.
        interval (tuple): Sampling interval (pixel).
        value_threshold (float): Density threshold.
        num_points (int): Approximate number of points.

    Returns:
        numpy.ndarray: (N, 3) coordinates
    """
    # x_min, x_max, y_min, y_max, z_min, z_max = bbox
    grid_shape = data.shape  # Shape of the 3D array (nx, ny, nz)

    # Generate grid coordinates
    x = np.arange(start=0, stop=grid_shape[0], step=interval)
    y = np.arange(start=0, stop=grid_shape[1], step=interval)
    z = np.arange(start=0, stop=grid_shape[2], step=interval)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten arrays for easier filtering
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    sampled_values = data[grid_x, grid_y, grid_z].ravel()
    # Apply threshold to select points
    filtered_points = grid_coords[sampled_values >= value_threshold]

    # Optionally, downsample the result to approximately `num_points`
    if num_points is not None and len(filtered_points) > num_points:
        indices = np.random.choice(len(filtered_points), num_points, replace=False)
        filtered_points = filtered_points[indices]

    return filtered_points


def unifrom_sample_with_value_threshold(data, interval, value_threshold):
    """
    Parameters:
        data (numpy.ndarray): 3D array.
        interval (tuple): Sampling interval (pixel).
        value_threshold (float): Density threshold.

    Returns:
        numpy.ndarray: (N,) values
        numpy.ndarray: (N, 3) coordinates
    """
    z_size, y_size, x_size = data.shape
    
    z = np.arange(z_size)
    y = np.arange(y_size)
    x = np.arange(x_size)
    
    interpolator = RegularGridInterpolator((z, y, x), data, method='linear')
    
    z_samples = np.arange(0, z_size-1e-9, interval)
    y_samples = np.arange(0, y_size-1e-9, interval)
    x_samples = np.arange(0, x_size-1e-9, interval)
    
    Z, Y, X = np.meshgrid(z_samples, y_samples, x_samples, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    sampled_values = interpolator(np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=-1))
    
    mask = sampled_values >= value_threshold
    filtered_values = sampled_values[mask]
    filtered_coords = points[mask]
    
    return filtered_values, filtered_coords


def downsample_mrc_images(
    src: ImageSource,
    new_box_size: int,
    out_fl: str,
    batch_size: int,
):
    """Downsample the images in a single particle stack into a new .mrcs file.

    This utlility function also simplifies handling each of the individual .mrcs files
    listed in a .txt or .star file into a new collection of downsampled .mrcs files.

    Arguments
    ---------
    src         A loaded particle image stack.
    new_box_size       The new downsampled box size in pixels.
    out_fl      The output .mrcs file name.
    batch_size  The batch size for processing images;
                useful for avoiding out-of-memory issues.
    chunk_size  If given, divide output into files each containing this many images.

    """
    if new_box_size > src.D:
        raise ValueError(
            f"New box size {new_box_size} cannot be larger "
            f"than the original box size {src.D}!"
        )

    old_apix = src.apix
    if old_apix is None:
        old_apix = 1.0

    new_apix = np.round(old_apix * src.D / new_box_size, 6)
    if isinstance(new_apix, Iterable):
        new_apix = tuple(set(new_apix))

        if len(new_apix) > 1:
            print(
                f"Found multiple A/px values in {src.filenames}, using the first one "
                f"found {new_apix[0]} for the output .mrcs!"
            )
        new_apix = new_apix[0]

    def downsample_transform(chunk: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
        """Downsample an image array by clipping Fourier frequencies."""

        start = int(src.D / 2 - new_box_size / 2)
        stop = int(src.D / 2 + new_box_size / 2)
        oldft = ht2_center(chunk)
        # oldft = fft.ht2_center(chunk)

        newft = oldft[:, start:stop, start:stop]
        # new_chunk = fft.iht2_center(newft)
        new_chunk = iht2_center(newft)

        return new_chunk

    header = MRCHeader.make_default_header(
        nz=src.n,
        ny=new_box_size,
        nx=new_box_size,
        Apix=new_apix,
        dtype=src.dtype,
        is_vol=False,
    )
    src.write_mrc(
        output_file=out_fl,
        header=header,
        transform_fn=downsample_transform,
        chunksize=batch_size,
    )


if __name__ == "__main__":

    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if use_cuda else 'cpu')
    # if use_cuda:
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #     print('Using cuda')

    parser = ArgumentParser(description="Extract particle info")
    
    parser.add_argument("--source_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--star_file", type=str, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--downsample_size", type=int, default=None)
    parser.add_argument("--apix", type=float, default=None)
    parser.add_argument("--consensus_map", type=str, default=None)
    parser.add_argument("--map_thres", type=float, default=None)
    parser.add_argument("--sample_interval", type=int, default=None)
    parser.add_argument("--particle_name", type=str, default='image_stack')
    parser.add_argument("-downsample_b",type=int,default=5000,help="Batch size for processing images (default: %(default)s)")
    parser.add_argument("--scale_min", type=float, default=0.5)
    parser.add_argument("--scale_max", type=float, default=1.0)

    group = parser.add_argument_group(description="Network params")
    group.add_argument("--gaussian_embedding_dim", type=int, default=32)
    group.add_argument("--qlayers", type=int, default=2)
    group.add_argument("--qdim", type=int, default=1024)
    group.add_argument("--zdim", type=int, default=10)
    group.add_argument("--gaussian_kdim", type=int, default=128)
    group.add_argument("--gaussian_klayers", type=int, default=3)
    group.add_argument("--feature_dim", type=int, default=256)
    group.add_argument("--feature_kdim", type=int, default=128)
    group.add_argument("--feature_klayers", type=int, default=2)
    
    group = parser.add_argument_group(description="Training script parameters")
    lp = ModelParams_EM(parser)
    op = OptimizationParams_EM(parser)
    pp = PipelineParams(parser)
    group.add_argument("--detect_anomaly", action="store_true", default=False)
    group.add_argument("--no_window", action="store_true")
    group.add_argument("--start_checkpoint", type=str, default=None)
    group.add_argument("--output", type=str, default=None)
    group.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args(sys.argv[1:])
    
    ### base info
    source_dir = args.source_dir
    store_data_dir = args.data_dir
    metafile = args.star_file
    gaussian_dir = "%s/gaussian_preprocess" % source_dir
    if not os.path.exists(gaussian_dir):
        os.makedirs(gaussian_dir)
    img_size = args.size
    img_down_size = args.downsample_size
    img_apix = args.apix

    print("Extracting particles and information")

    ### extract orientations, ctfs
    rots, trans = parse_pose_star(metafile, img_size, img_apix)
    if img_down_size:
        trans *= img_down_size / img_size
    rots = rots.reshape(-1, 9)
    orientations = np.concatenate([rots, trans], axis=1)
    ctfs = parse_ctf_star(metafile, img_size, img_apix)
    save_ctf = "%s/ctf.npy" % gaussian_dir
    save_pose = "%s/poses.npy" % gaussian_dir
    np.save(save_ctf, ctfs)
    np.save(save_pose, orientations)

    ### downsample particle images
    src = ImageSource.from_file(metafile, lazy=True, indices=None, datadir=args.data_dir)
    if args.downsample_size > src.D:
        raise ValueError(
            f"New box size {args.downsample_size=} can't be larger "
            f"than the original box size {src.D}!"
        )
    if args.downsample_size % 2 != 0:
        raise ValueError(f"New box size {args.downsample_size=} is not even!")
    downsample_mrc_images(src, args.downsample_size, os.path.join(gaussian_dir, args.particle_name+'.mrcs'), args.downsample_b)

    with mrcfile.open(os.path.join(gaussian_dir, args.particle_name+'.mrcs'), 'r') as mrc:
        downsample_particle = mrc.data
        std = np.std(downsample_particle)
        # print(std)

    print("Sampling initial Gaussians")

    ### sample consensus map
    if not os.path.isabs(args.consensus_map):
        init_map = "%s/%s" % (source_dir, args.consensus_map)
    else:
        init_map = args.consensus_map

    with mrcfile.open(init_map, 'r') as mrc:
        vol = mrc.data 
    density_thres = args.map_thres
    density_mask = vol > density_thres
    offOrigin = np.array([0, 0, 0])
    dVoxel = np.array([img_apix, img_apix, img_apix])
    real_size = img_size * img_apix
    sVoxel = np.array([real_size, real_size, real_size])
    interval = args.sample_interval
    sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    sampled_densities = vol[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ]

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    save_gauss = "%s/gaussians_%dinter.npy" % (gaussian_dir, interval)
    np.save(save_gauss, out)

    print("Generating configure file")

    ### configure file
    cfg_file = "%s/cfg.json" % gaussian_dir
    meta_data = {}
    meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    scanner_cfg = {}

    ### gaussian param
    scanner_cfg['mode'] = "parallel"
    scanner_cfg['filter'] = None
    scanner_cfg['DSD'] = 7.0
    scanner_cfg['DSO'] = 5.0
    real_size = img_size * img_apix
    scanner_cfg['nDetector'] = [img_size, img_size]
    scanner_cfg['sDetector'] = [real_size, real_size]
    scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    scanner_cfg['offOrigin'] = [0, 0, 0]
    scanner_cfg['offDetector'] = [0, 0, 0]
    scanner_cfg['accuracy'] = 0.5
    scanner_cfg['ctf'] = True
    scanner_cfg['invert_contrast'] = True
    scanner_cfg['pixelSize'] = img_apix
    scanner_cfg['scale_min'] = args.scale_min / img_size 
    scanner_cfg['scale_max'] = args.scale_max / img_size 
    scanner_cfg['std'] = float(std)

    ### network param
    scanner_cfg['gaussian_embedding_dim'] = args.gaussian_embedding_dim
    scanner_cfg['qlayers'] = args.qlayers 
    scanner_cfg['qdim'] = args.qdim
    scanner_cfg['zdim'] = args.zdim
    scanner_cfg['gaussian_kdim'] = args.gaussian_kdim
    scanner_cfg['gaussian_klayers'] = args.gaussian_klayers
    scanner_cfg['feature_dim'] = args.feature_dim
    scanner_cfg['feature_kdim'] = args.feature_kdim
    scanner_cfg['feature_klayers'] = args.feature_klayers

    meta_data['cfg'] = scanner_cfg
    with open(cfg_file, "w") as f:
        json.dump(meta_data, f)
    
    ###homo_recon_part
    
    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    window_flag = not args.no_window
    dataset = Particles(gaussian_dir, args.particle_name, save_pose, save_ctf, save_gauss, cfg_file, window=window_flag)

    args.position_lr_max_steps = args.epoch * dataset.particle_num
    args.density_lr_max_steps =  args.epoch * dataset.particle_num
    args.scaling_lr_max_steps =  args.epoch * dataset.particle_num
    args.rotation_lr_max_steps = args.epoch * dataset.particle_num
    args.densify_until_iter =  args.epoch * dataset.particle_num
    args.densify_grad_threshold = 5.0e-5
    args.densify_scale_threshold = scanner_cfg['scale_max']
    args.max_scale = scanner_cfg['scale_max'] + 0.00025
    args.contribution_prune_ratio = 0.1

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
    
    os.system('cp ' + osp.join(gaussian_dir, f"point_cloud_{args.particle_name}", f"epoch_{args.epoch - 1}", "point_cloud.ply") + ' ' + gaussian_dir)
