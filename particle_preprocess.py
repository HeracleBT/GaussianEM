import numpy as np
import torch
from typing import Optional
import mrcfile
from scipy.interpolate import RegularGridInterpolator
import pickle
from data.mrcfile_em import MRCHeader
from data.relion import ImageSource
import json

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


def calculate_dose_weights(ntilts, D, pixel_size, dose_per_A2_per_tilt, voltage):
    '''
    code adapted from Grigorieff lab summovie_1.0.2/src/core/electron_dose.f90
    see also Grant and Grigorieff, eLife (2015) DOI: 10.7554/eLife.06980
    assumes even-sized FFT (i.e. non-ht-symmetrized, DC component is bottom-right of central 4 px)
    '''
    cumulative_doses = dose_per_A2_per_tilt * np.arange(1, ntilts+1)
    dose_weights = np.zeros((ntilts, D, D))
    fourier_pixel_sizes = 1.0 / (np.array([D, D]))  # in units of 1/px
    box_center_indices = np.array([D, D]) // 2
    critical_dose_at_dc = 0.001 * (2 ** 31) # shorthand way to ensure dc component is always weighted ~1
    voltage_scaling_factor = 1.0 if voltage == 300 else 0.8  # 1.0 for 300kV, 0.8 for 200kV microscopes

    for k, dose_at_end_of_tilt in enumerate(cumulative_doses):

        for j in range(D):
            y = ((j - box_center_indices[1]) * fourier_pixel_sizes[1])

            for i in range(D):
                x = ((i - box_center_indices[0]) * fourier_pixel_sizes[0])

                if ((i, j) == box_center_indices).all():
                    spatial_frequency_critical_dose = critical_dose_at_dc
                else:
                    spatial_frequency = np.sqrt(x ** 2 + y ** 2) / pixel_size  # units of 1/A
                    spatial_frequency_critical_dose = (0.24499 * spatial_frequency ** (
                        -1.6649) + 2.8141) * voltage_scaling_factor  # eq 3 from DOI: 10.7554/eLife.06980

                dose_weights[k, j, i] = np.exp((-0.5 * dose_at_end_of_tilt) / spatial_frequency_critical_dose)  # eq 5 from DOI: 10.7554/eLife.06980

    assert dose_weights.min() >= 0.0
    assert dose_weights.max() <= 1.0
    return dose_weights


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


def compute_full_ctf(D, Nimg, ctf_params):
    freqs = np.arange(-D / 2, D / 2) / (ctf_params.Apix * D)
    x0, x1 = np.meshgrid(freqs, freqs)
    freqs = torch.tensor(np.stack([x0.ravel(), x1.ravel()], axis=1)).cuda()
    ctf = compute_ctf(
        freqs,
        torch.Tensor([ctf_params.dfu]).reshape(-1, 1),
        torch.Tensor([ctf_params.dfv]).reshape(-1, 1),
        torch.Tensor([ctf_params.ang]).reshape(-1, 1),
        torch.Tensor([ctf_params.kv]).reshape(-1, 1),
        torch.Tensor([ctf_params.cs]).reshape(-1, 1),
        torch.Tensor([ctf_params.wgh]).reshape(-1, 1),
        phase_shift = torch.Tensor([ctf_params.ps]).reshape(-1, 1),
        bfactor = torch.Tensor([ctf_params.b]).reshape(-1, 1),
    )
    ctf = ctf.reshape((D, D))
    df = np.stack([np.ones(Nimg) * ctf_params.dfu, np.ones(Nimg) * ctf_params.dfv], axis=1)
    return ctf, df


def standardize(images):
    images = (images - np.mean(images, axis=(1, 2)).reshape(-1, 1, 1)) / np.std(images, axis=(1, 2)).reshape(-1, 1, 1)
    return images


def sign_flip(chunk: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    """data * -1, convert bright to dark."""

    return chunk * -1


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
    x = np.arange(start=0, stop=grid_shape[0], step=interval[0])
    y = np.arange(start=0, stop=grid_shape[1], step=interval[1])
    z = np.arange(start=0, stop=grid_shape[2], step=interval[2])
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


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


    ### cryobench IgG-1D
    source_dir = "/home/data_8T/EMPAIR/cryobench/IgG-1D"
    store_data_dir = "%s/images/snr0.01" % source_dir
    metafile = "%s/snr0.01.star" % store_data_dir
    ctf_dir = "%s/ctf.pkl" % source_dir
    pose_dir = "%s/pose.pkl" % source_dir
    gaussian_dir = "%s/gaussian_preprocess" % source_dir
    img_size = 128
    img_apix = 3.0

    with open(ctf_dir, 'rb') as f:
        ctf = pickle.load(f)

    with open(pose_dir, 'rb') as f:
        poses = pickle.load(f)

    rots, trans = poses[0], poses[1]
    trans *= img_size
    rots = rots.reshape(-1, 9)
    orientations = np.concatenate([rots, trans], axis=1)
    save_ctf = "%s/ctf.npy" % gaussian_dir
    save_pose = "%s/poses.npy" % gaussian_dir
    np.save(save_ctf, ctf)
    np.save(save_pose, orientations)

    image_subset1 = "%s/img_stack.mrcs" % gaussian_dir
    src1 = ImageSource.from_file(metafile, lazy=True, datadir=store_data_dir)
    print(src1.shape)
    
    header = MRCHeader.make_default_header(
        nz=src1.n,
        ny=img_size,
        nx=img_size,
        Apix=img_apix,
        dtype=src1.dtype,
        is_vol=False,
    )
    src1.write_mrc(
        output_file=image_subset1,
        header=header,
        transform_fn=None,
        chunksize=5000,
    )


    init_map = "%s/vols/128_org/049.mrc" % source_dir
    with mrcfile.open(init_map, 'r') as mrc:
        vol = mrc.data 

    density_thres = 0.0228
    density_mask = vol > density_thres
    offOrigin = np.array([0, 0, 0])
    dVoxel = np.array([img_apix, img_apix, img_apix])
    real_size = img_size * img_apix
    sVoxel = np.array([real_size, real_size, real_size])
    interval = (1, 1, 1)
    sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    print(sampled_indices.shape)
    sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    sampled_densities = vol[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ]

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    save_path = "%s/gaussians_1inter_049.npy" % gaussian_dir
    np.save(save_path, out)


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
    scanner_cfg['scale_min'] = 0.5 / img_size
    scanner_cfg['scale_max'] = 1.0 / img_size
    scanner_cfg['std'] = 8.7975

    ### network param
    scanner_cfg['gaussian_embedding_dim'] = 32
    scanner_cfg['qlayers'] = 3
    scanner_cfg['qdim'] = 1024
    scanner_cfg['zdim'] = 10
    scanner_cfg['gaussian_kdim'] = 128
    scanner_cfg['gaussian_klayers'] = 3
    scanner_cfg['feature_dim'] = 256
    scanner_cfg['feature_kdim'] = 128
    scanner_cfg['feature_klayers'] = 2

    meta_data['cfg'] = scanner_cfg

    with open(cfg_file, "w") as f:
        json.dump(meta_data, f)
