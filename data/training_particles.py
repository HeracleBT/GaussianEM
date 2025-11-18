import os.path as osp
import torch
import sys
import numpy as np
import mrcfile
import starfile
import json

sys.path.append("./")
from model.gaussian_model_insight import GaussianModel
from utils.general_utils import t2a
from utils.utils_em import circular_mask, R_from_relion, relion_angle_to_matrix
from data.dataset_em import ImageDataset


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

        self.mask_dir = "%s/%s" % (self.path, "mask_indices.npy")
        if osp.exists(self.mask_dir):
            print("mask operation")
            self.mask_indices = np.load(self.mask_dir)[:, None].astype(np.float32)
            self.mask_indices = torch.tensor(self.mask_indices).cuda()
        else:
            self.mask_indices = None
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
            vol_pred_mrc = osp.join(point_cloud_path, "vol_pred.mrc")
            with mrcfile.new(vol_pred_mrc, overwrite=True) as mrc:
                mrc.set_data(t2a(vol_pred).astype(np.float32))
                mrc.voxel_size = self.pixel_size
