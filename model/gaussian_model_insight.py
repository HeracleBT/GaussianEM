import os
import sys
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.append("./")

from simple_knn._C import distCUDA2
from r2_gaussian.utils.system_utils import mkdir_p
from r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    inverse_tanh,
    strip_symmetric,
    build_scaling_rotation,
)
from scipy.spatial import cKDTree

class GaussianModel:
    def setup_functions(self, scale_bound=None, max_density=None):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if scale_bound is not None:
            # self.size_scale = 64 * 3.0
            # self.scaling_activation = (
            #     lambda x: (torch.tanh(x) * 0.5 + 1.0)
            # )
            # self.scaling_inverse_activation = lambda x: inverse_tanh(
            #     (x - 1.0) * 2
            # )
            # self.scale_bound = np.array([0.5, 1.5]) / self.size_scale
            scale_min_bound, scale_max_bound = scale_bound
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound) + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound)) + 1e-8
            )
            self.scale_bound = scale_bound
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
            self.scale_bound = None
        self.covariance_activation = build_covariance_from_scaling_rotation

        # if max_density is not None:
        #     self.density_activation = lambda x: torch.tanh(x) + 1.0
        #     self.density_inverse_activation = lambda x: inverse_tanh(x - 1.0)
        #     self.max_density = max_density
        # else:
        #     self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        #     self.density_inverse_activation = inverse_softplus
        #     self.max_density = None

        if max_density is not None:
            self.density_activation = (
                lambda x: torch.clamp_max(torch.nn.functional.softplus(x), max_density)
            )
            self.density_inverse_activation = lambda x: inverse_softplus(
                torch.clamp_max(x, max_density)
            )
            self.max_density = max_density
        else:
            self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
            self.density_inverse_activation = inverse_softplus
            self.max_density = None

        # self.xyz_activation = torch.nn.Tanh()
        # self.xyz_inverse_activation = inverse_tanh

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, scale_bound=None, max_density=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions(scale_bound, max_density)

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # return self.scaling_activation(self._scaling.repeat(1, 3)) / self.size_scale
        return self.scaling_activation(self._scaling.repeat(1, 3))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(
            "Initialize gaussians from {} estimated points".format(
                fused_point_cloud.shape[0]
            )
        )
        dist2 = distCUDA2(fused_point_cloud)
        filter_indices = dist2 < torch.mean(dist2) * 3
        dist2 = dist2[filter_indices]
        fused_point_cloud = fused_point_cloud[filter_indices]
        # fused_point_cloud = self.xyz_inverse_activation(fused_point_cloud)
        density = torch.tensor(density)
        if self.max_density is not None:
            density = torch.clamp_max(density, self.max_density * 0.9)
        fused_density = (
            self.density_inverse_activation(density).float().cuda()
        )
        fused_density = fused_density[filter_indices]

        # print(torch.sqrt(dist2))
        
        # print(self.scale_bound)
        # sys.exit()
        if self.scale_bound is not None:
            min_val = self.scale_bound[0] ** 2
            max_val = (self.scale_bound[0] + self.scale_bound[1]) ** 2 / 4
            dist2 = torch.clamp(
                dist2,
                min = min_val,
                max = max_val
            )
        # print(torch.max(torch.sqrt(dist2)), torch.min(torch.sqrt(dist2)))
        # print(torch.mean(torch.sqrt(dist2)))
        # print(torch.quantile(torch.sqrt(dist2), 0.9))
        # print(self.scale_bound)
        # print(torch.sqrt(dist2))
        # print(self.scaling_inverse_activation(torch.sqrt(dist2) * self.size_scale)[..., None])
        # sys.exit()
        # print(torch.where(torch.isnan(dist2)))
        # print(dist2[torch.where(torch.isnan(dist2))])
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None]
        # scales = self.scaling_inverse_activation(torch.sqrt(dist2) * self.size_scale)[..., None]

        # print(dist2[torch.where(torch.isnan(self.scaling_inverse_activation(torch.sqrt(dist2))))])
        # scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(
        #     1, 3
        # )
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer = torch.optim.Adam(l)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )


    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr


    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        prune_mask = torch.logical_or(self._xyz.isnan(), self._density.isnan())[:, 0]

        xyz = self._xyz[~prune_mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        densities = self._density[~prune_mask].detach().cpu().numpy()
        scale = self._scaling[~prune_mask].detach().cpu().numpy()
        rotation = self._rotation[~prune_mask].detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, densities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        densities = np.asarray(plydata.elements[0]["density"])[..., np.newaxis]

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._density = nn.Parameter(
            torch.tensor(densities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False)
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                if name == "rotation":
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(False))
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                if group["name"] == "rotation":
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].detach().requires_grad_(False))
                    )
                else:
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "rotation":
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].detach().requires_grad_(False))
                    )
                else:
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                if group["name"] == "rotation":
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(False)
                    )
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "rotation":
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(False)
                    )
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_scaling,
        new_rotation,
        new_max_radii2D,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )
        # selected_pts_mask = torch.logical_and(
        #     selected_pts_mask,
        #     torch.min(self.get_scaling, dim=1).values > 0.005,
        # )

        # print(padded_grad[selected_pts_mask])
        # print(torch.sum(selected_pts_mask), " points split")

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # samples = (torch.rand((stds.size(0), 3), device="cuda") * 2 - 1) * stds
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )[..., :1]
        # new_scaling = self.scaling_inverse_activation(
        #     self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.6 * N)
        # )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        # new_density = self.density_inverse_activation(
        #     self.get_density[selected_pts_mask].repeat(N, 1) / (N / np.sqrt(2))
        # )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, densify_scale_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )

        # print(torch.sum(selected_pts_mask), " points clone111")
        # temp_norm = torch.norm(grads, dim=-1)
        # print(temp_norm[selected_pts_mask])
        # print(grad_threshold)

        # selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)

        # print(torch.sum(selected_pts_mask), " points clone222")

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        # print(torch.sum(selected_pts_mask), " points clone")
        # print(grads[selected_pts_mask])
        # print(grads_abs[selected_pts_mask])
        # if torch.sum(selected_pts_mask) > 0:
        #     temp_norm = torch.norm(grads, dim=-1)
        #     # temp_abs_norm = torch.norm(grads_abs, dim=-1)
        #     print(temp_norm[selected_pts_mask])

        new_xyz = self._xyz[selected_pts_mask]
        new_densities = self._density[selected_pts_mask]
        # new_densities = self.density_inverse_activation(
        #     self.get_density[selected_pts_mask] * 0.5
        # )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # print(torch.mean(grads), torch.std(grads))

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, grads_abs, Q, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, grads_abs, Q, densify_scale_threshold)

        # Prune gaussians with too small density
        prune_mask = (self.get_density < min_density).squeeze()

        # print(torch.sum(prune_mask), "prune poinst")
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

            # print(torch.mean(self.get_scaling, dim=1), max_scale)
            # print(torch.sum(big_points_ws))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(
            self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1


class SimpleGaussian:
    def __init__(self, scale_bound=None, activate=None, max_density=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self.max_radii2D = torch.empty(0)

        if 'density' in activate:
            # if max_density is not None:
            #     self.density_activation = lambda x: torch.tanh(x) + 1.0
            #     self.density_inverse_activation = lambda x: inverse_tanh(x - 1.0)
            #     self.max_density = max_density
            # else:
            #     self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
            #     self.density_inverse_activation = inverse_softplus
            #     self.max_density = None
            if max_density is not None:
                self.density_activation = (
                    lambda x: torch.clamp_max(torch.nn.functional.softplus(x), max_density)
                )
                self.density_inverse_activation = lambda x: inverse_softplus(
                    torch.clamp_max(x, max_density)
                )
                self.max_density = max_density
            else:
                self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
                self.density_inverse_activation = inverse_softplus
                self.max_density = None
        else:
            self.density_activation = None

        if 'scaling' in activate:
            if scale_bound is not None:
                # self.size_scale = 64 * 3.0
                # self.scaling_activation = (
                #     lambda x: (torch.tanh(x) * 0.5 + 1.0)
                # )
                # self.scaling_inverse_activation = lambda x: inverse_tanh(
                #     (x - 1.0) * 2
                # )
                # self.scale_bound = np.array([0.5, 1.5]) / self.size_scale
                scale_min_bound, scale_max_bound = scale_bound
                self.scaling_activation = (
                    lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound) + scale_min_bound
                )
                self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                    torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound)) + 1e-8
                )
                self.scale_bound = scale_bound
            else:
                self.scaling_activation = torch.exp
                self.scaling_inverse_activation = torch.log
                self.scale_bound = None
        else:
            self.scaling_activation = None
            self.scale_bound = None

        if 'position' in activate:
            # self.xyz_activation = torch.nn.Tanh()
            # self.xyz_inverse_activation = inverse_tanh
            self.xyz_activation = True
        else:
            self.xyz_activation = None

    @property
    def get_scaling(self):
        if self.scaling_activation is None:
            return self._scaling.repeat(1, 3)
        else:
            return self.scaling_activation(self._scaling.repeat(1, 3))

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        # if self.xyz_activation is None:
        #     return self._xyz
        # else:
        #     return self.xyz_activation(self._xyz)
        return self._xyz

    @property
    def get_density(self):
        if self.density_activation is None:
            return self._density
        else:
            return self.density_activation(self._density)


    def create_from_Gaussian(self, gaussians):
        xyz = gaussians.get_xyz
        rots = gaussians.get_rotation

        if self.density_activation is not None:
            density_requires_grad = True
            densities = gaussians._density
        else:
            density_requires_grad = False
            densities = gaussians.get_density

        if self.scaling_activation is not None:
            scaling_requires_grad = True
            scales = gaussians._scaling
        else:
            scaling_requires_grad = False
            scales = gaussians.get_scaling
        
        if self.xyz_activation is not None:
            xyz_requires_grad = True
            # xyz = gaussians._xyz
        else:
            xyz_requires_grad = False
            # xyz = gaussians.get_xyz

        self._xyz = nn.Parameter(xyz.clone().detach(), requires_grad=xyz_requires_grad)
        self._density = nn.Parameter(densities.clone().detach(), requires_grad=density_requires_grad)
        self._scaling = nn.Parameter(scales.clone().detach(), requires_grad=scaling_requires_grad)
        self._rotation = nn.Parameter(rots.clone().detach(), requires_grad=False)


    def create_from_values(self, xyz, densities, scales, rots):
        if self.density_activation is not None:
            density_requires_grad = True
        else:
            density_requires_grad = False

        if self.scaling_activation is not None:
            scaling_requires_grad = True
        else:
            scaling_requires_grad = False

        if self.xyz_activation is not None:
            xyz_requires_grad = True
        else:
            xyz_requires_grad = False

        self._xyz = nn.Parameter(xyz.clone().detach(), requires_grad=xyz_requires_grad)
        self._density = nn.Parameter(densities.clone().detach(), requires_grad=density_requires_grad)
        self._scaling = nn.Parameter(scales.clone().detach(), requires_grad=scaling_requires_grad)
        self._rotation = nn.Parameter(rots.clone().detach(), requires_grad=False)

