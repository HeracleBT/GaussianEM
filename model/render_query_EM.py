import sys
import torch
import numpy as np
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

sys.path.append("./")
from model.gaussian_model_insight import GaussianModel

def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    scaling_modifier=1.0,
    reorder=False
):
    """
    Query a volume with voxelization.
    """
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=False,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    if reorder:
        means3D = pc.get_xyz.detach()
        means3D = torch.flip(means3D, [1])
    else:
        means3D = pc.get_xyz
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    vol_pred, radii = voxelizer(
        means3D=means3D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def render(orientation, gaussians, img_size, pipe=None, record_transmittance=False):
    rot, tran = orientation
    world_view_transform = np.eye(4, dtype=np.float32)
    world_view_transform[:3, :3] = rot
    world_view_transform = torch.from_numpy(world_view_transform).transpose(0, 1).cuda()
    projection_matrix = torch.eye(4)
    projection_matrix[0, 3] = -tran[0]
    projection_matrix[1, 3] = -tran[1]
    projection_matrix = projection_matrix.transpose(0, 1).cuda()
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    world_view_transform[3, 2] = 5
    tanfovx = 1.0
    tanfovy = 1.0
    scaling_modifier = 1.0

    if pipe is None:
        debug_flag = False
        compute_cov3D_flag = False
    else:
        debug_flag = pipe.debug
        compute_cov3D_flag = pipe.compute_cov3D_python

    raster_settings = GaussianRasterizationSettings(
        image_height=img_size,
        image_width=img_size,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        campos=camera_center,
        prefiltered=False,
        record_transmittance=record_transmittance,
        mode=0,
        debug=debug_flag,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    screenspace_points = (
        torch.zeros_like(
            gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = gaussians.get_xyz
    means2D = screenspace_points
    density = gaussians.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if compute_cov3D_flag:
        cov3D_precomp = gaussians.get_covariance(scaling_modifier)
    else:
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    output = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if record_transmittance:
        transmittance_sum, num_covered_pixels, radii = output
        transmittance = transmittance_sum / (num_covered_pixels + 1e-6)
        return transmittance
    else:
        rendered_image, radii = output
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
