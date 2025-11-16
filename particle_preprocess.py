import numpy as np
import torch
from typing import Optional
import mrcfile
import matplotlib.pyplot as plt
from utils_em import R_from_relion, expmap
from relion import Relionfile, ImageSource
import pickle
import json
from utils_em import spherical_mask
from scipy.interpolate import RegularGridInterpolator

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


def add_ctf(particles, ctf):
    particles = np.array(
        [np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x))) for x in particles]
    )
    particles *= np.array(ctf.cpu())
    del ctf
    particles = np.array(
        [
            np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x))).astype(np.float32)
            for x in particles
        ]
    )
    return particles


def add_ctf_torch(particles, ctf):
    batch_prtl = torch.from_numpy(particles.copy()).cuda()
    batch_prtl = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(batch_prtl, dim=(-2, -1))), dim=(-2, -1))
    # batch_prtl = batch_prtl.real - batch_prtl.imag
    batch_prtl *= ctf
    batch_prtl = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(batch_prtl, dim=(-2, -1))), dim=(-2, -1))
    # batch_prtl = batch_prtl.real - batch_prtl.imag

    # batch_prtl = torch.fft.fftshift(torch.fft.fft2(batch_prtl), dim=(-1, -2))
    # batch_prtl = torch.multiply(batch_prtl, ctf)
    # batch_prtl = torch.fft.ifft2(torch.fft.ifftshift(batch_prtl, dim=(-1, -2)))
    return batch_prtl


def add_noise(particles, snr):
    std = np.std(particles)
    sigma = std / np.sqrt(snr)
    noise = np.random.normal(0, sigma, particles.shape)
    return particles + noise


def add_noise_torch(particles, snr):
    std = torch.std(particles)
    sigma = std / np.sqrt(snr)
    noise = torch.normal(0, sigma, particles.shape)
    return particles + noise


def standardize(images):
    images = (images - np.mean(images, axis=(1, 2)).reshape(-1, 1, 1)) / np.std(images, axis=(1, 2)).reshape(-1, 1, 1)
    return images


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


def parse_pose_csparc(input, abinit=False, hetrefine=False):
    data = np.load(input)
    #### view the first row
    # for i in range(len(data.dtype)):
    #     print(i, data.dtype.names[i], data[0][i])

    # for i in range(50):
    #     if data[i][2] != i:
    #         print(i, data[i][2])

    if abinit:
        RKEY = "alignments_class_0/pose"
        TKEY = "alignments_class_0/shift"
    else:
        RKEY = "alignments3D/pose"
        TKEY = "alignments3D/shift"
    
    rot = np.array([x[RKEY] for x in data])
    rot = torch.tensor(rot)
    rot = expmap(rot)
    rot = rot.cpu().numpy()
    rot = np.array([x.T for x in rot])
    trans = np.array([x[TKEY] for x in data])
    if hetrefine:
        trans *= 2
    return rot, trans


def simulation_ctf_noise(data_path, particles_file, output_file, ctf_setting, snr):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    particles_dir = "%s/%s.mrcs" % (data_path, particles_file)
    with mrcfile.open(particles_dir) as mrc:
        particles = mrc.data
    Nimg, D, D = particles.shape
    ctf, defocus_list = compute_full_ctf(D, Nimg, ctf_setting)
    particles = standardize(particles)

    corrupt_particles = add_ctf(particles, ctf)
    noised_particles = add_noise(corrupt_particles, snr)

    corrupted_stack = "%s/%s.mrcs" % (data_path, output_file)
    with mrcfile.new(corrupted_stack) as mrc:
        noised_particles = noised_particles.astype(np.float32)
        mrc.set_data(noised_particles)


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
    
    # 创建原始网格
    z = np.arange(z_size)
    y = np.arange(y_size)
    x = np.arange(x_size)
    
    # 创建插值函数
    interpolator = RegularGridInterpolator((z, y, x), data, method='linear')
    
    # 计算采样点
    z_samples = np.arange(0, z_size-1e-9, interval)
    y_samples = np.arange(0, y_size-1e-9, interval)
    x_samples = np.arange(0, x_size-1e-9, interval)
    
    # 创建采样点网格
    Z, Y, X = np.meshgrid(z_samples, y_samples, x_samples, indexing='ij')
    # 调整坐标顺序为x, y, z
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    # 进行插值采样
    sampled_values = interpolator(np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=-1))
    
    # 应用阈值过滤
    mask = sampled_values >= value_threshold
    filtered_values = sampled_values[mask]
    filtered_coords = points[mask]
    
    return filtered_values, filtered_coords


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    ### simulation
    # # data_path = "dataset/simulation"
    # # data_path = "dataset/simulation_halfmap"
    # # data_path = "/home/data_8T/EMPAIR/cryobench/IgG-1D/simulated/"
    # data_path = "/home/data_8T/EMPAIR/simulation"
    # # particles_dir = "%s/state_20_clean.mrcs" % data_path
    # # particles_dir = "%s/state_36_clean.mrcs" % data_path
    # # particles_dir = "%s/039_clean.mrcs" % data_path
    # particles_dir = "%s/state_30_clean.mrcs" % data_path
    # with mrcfile.open(particles_dir) as mrc:
    #     particles = mrc.data
    # # load particles
    # Nimg, D, D = particles.shape
    # # ctf_setting = ctf_params()
    # # ctf, defocus_list = compute_full_ctf(D, Nimg, ctf_setting)
    # particles = standardize(particles)

    # # image = particles[0]
    # # print(np.max(image), np.min(image))
    # # print(np.mean(image), np.std(image))
    # # image_f = np.fft.fft2(image)
    # # image_s = np.fft.fftshift(image)
    # # image_sf = np.fft.fft2(image_s)
    # # image_fs = np.fft.fftshift(image_f)
    # # image_sf_i = np.fft.ifft2(image_sf)
    # # image_fs_i = np.fft.ifft2(image_fs)
    # # # axes[0].imshow(ctf.cpu(), cmap='gray')
    # # fig, axes = plt.subplots(1, 5)
    # # axes[0].imshow(image, cmap='gray')
    # # axes[1].imshow(image_s, cmap='gray')
    # # axes[2].imshow(image_f.real, cmap='gray')
    # # axes[3].imshow(image_sf_i.real, cmap='gray')
    # # axes[4].imshow(image_fs_i.real, cmap='gray')
    # # plt.show()

    # # print(ctf.shape)
    # # print(particles.shape)


    # # corrupt_particles_torch = add_ctf_torch(particles, ctf)
    # # corrupt_particles_torch = corrupt_particles_torch.cpu().data.numpy().astype(np.float32)
    # # # noised_particles_torch = add_noise_torch(corrupt_particles_torch, 0.1)
    # # # noised_particles_torch *= -1
    # # # noised_particles_torch = noised_particles_torch.cpu().data.numpy().astype(np.float32)


    # # print(ctf.shape)
    # # corrupt_particles = add_ctf(particles, ctf)
    # snr = 0.05
    # # noised_particles = add_noise(corrupt_particles, snr)
    # # noised_particles *= -1  #  it's common for EM processing
    # noised_particles = add_noise(particles, snr)
    # # noised_particles = standardize(noised_particles)

    # raw_img = particles[0]
    # corrupt_img = corrupt_particles[0]
    # # noised_img = noised_particles[0]
    # corrupt_img_torch = corrupt_particles_torch[0]
    # # noised_img_torch = noised_particles_torch[0]
    # fig, axes = plt.subplots(1, 3)
    # axes[0].imshow(raw_img, cmap='gray')
    # axes[1].imshow(corrupt_img, cmap='gray')
    # # axes[2].imshow(noised_img, cmap='gray')
    # axes[2].imshow(corrupt_img_torch, cmap='gray')
    # # axes[1].imshow(noised_img, cmap='gray')
    # # axes[2].imshow(noised_img_torch, cmap='gray')
    # plt.show()


    # # print(ctf.shape)
    # ctf_output = "%s/simulation_ctf.npy" % data_path
    # ctf = ctf.cpu().data.numpy()
    # np.save(ctf_output, ctf)


    # # corrupted_stack = "%s/state_20_ctf_0.1_std.mrcs" % data_path
    # # corrupted_stack = "%s/state_36_ctf_0.1_std.mrcs" % data_path
    # # corrupted_stack = "%s/039.mrcs" % data_path
    # corrupted_stack = "%s/state_30.mrcs" % data_path
    # with mrcfile.new(corrupted_stack) as mrc:
    #     noised_particles = noised_particles.astype(np.float32)
    #     mrc.set_data(noised_particles)


    # img_size = 192
    # img_apix = 1.0
    # init_map = "%s/state_20.mrc" % data_path
    # with mrcfile.open(init_map, 'r') as mrc:
    #     vol = mrc.data 
    # density_thres = 1.39
    # density_mask = vol > density_thres
    # offOrigin = np.array([0, 0, 0])
    # dVoxel = np.array([img_apix, img_apix, img_apix])
    # real_size = img_size * img_apix
    # sVoxel = np.array([real_size, real_size, real_size])
    # interval = (1, 1, 1)
    # # interval = (2, 2, 2)
    # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # print(sampled_indices.shape)
    # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    # sampled_densities = vol[
    #     sampled_indices[:, 0],
    #     sampled_indices[:, 1],
    #     sampled_indices[:, 2],
    # ]

    # # sampled_densities = sampled_densities / sampled_densities.std()

    # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # save_path = "%s/gaussians_1inter.npy" % data_path
    # np.save(save_path, out)


    # cfg_file = "%s/cfg_scale_0.25_0.75.json" % data_path
    # meta_data = {}
    # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # scanner_cfg = {}
    # scanner_cfg['mode'] = "parallel"
    # scanner_cfg['filter'] = None
    # scanner_cfg['DSD'] = 7.0
    # scanner_cfg['DSO'] = 5.0
    # real_size = img_size * img_apix
    # scanner_cfg['nDetector'] = [img_size, img_size]
    # scanner_cfg['sDetector'] = [real_size, real_size]
    # scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # scanner_cfg['offOrigin'] = [0, 0, 0]
    # scanner_cfg['offDetector'] = [0, 0, 0]
    # scanner_cfg['accuracy'] = 0.5
    # scanner_cfg['ctf'] = True
    # scanner_cfg['invert_contrast'] = True
    # scanner_cfg['pixelSize'] = img_apix

    # scanner_cfg['scale_min'] = 0.25 / real_size
    # scanner_cfg['scale_max'] = 0.75 / real_size
    # meta_data['cfg'] = scanner_cfg

    # with open(cfg_file, "w") as f:
    #     json.dump(meta_data, f)


    # ctf_dir = "%s/ctf.pkl" % data_path
    # pose_dir = "%s/pose.pkl" % data_path

    # with open(ctf_dir, 'rb') as f:
    #     ctf = pickle.load(f)

    # ctf_output = "%s/ctfs.npy" % data_path
    # np.save(ctf_output, ctf)

    # with open(pose_dir, 'rb') as f:
    #     poses = pickle.load(f)

    # rots, trans = poses[0], poses[1]
    # trans *= img_size
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)
    # # print(orientations[0])
    # pose_output = "%s/poses.npy" % data_path
    # np.save(pose_output, orientations)


    # ### preprocess
    # # source_dir = "dataset/10180"
    # # source_dir = "/home/data_8T/EMPAIR/10005"
    # # source_dir = "/home/data_8T/EMPAIR/10076"
    # # source_dir = "/home/data_8T/EMPAIR/10180"
    # # source_dir = "/home/data_8T/EMPAIR/cryobench/Ribosembly"
    # source_dir = "/home/data_8T/EMPAIR/10841"
    # # source_dir = "/home/data_8T/EMPAIR/10059"
    # # metafile = "%s/consensus_data.star" % source_dir
    # # metafile = "%s/data/particles/tv1_relion_data.star" % source_dir
    # metafile = "%s/L17all.star" % source_dir
    # # metafile = "%s/data/particles/particles.star" % source_dir
    # # metafile = "%s/cryosparc_P15_J24_particles.cs" % source_dir
    # # metafile = "%s/cryosparc_particles.cs" % source_dir
    # # metafile = "%s/cryosparc_P25_J6_005_particles.cs" % source_dir
    # # metafile = "%s/cryosparc_P17_J7_004_particles.cs" % source_dir
    # # metafile = "%s/snr0.01.star" % source_dir
    # # store_data_dir = "/home/data_8T/EMPAIR/10180/data"
    # # store_data_dir = "/home/data_8T/EMPAIR/10005/data/particles"
    # # store_data_dir = "/home/data_8T/EMPAIR/10076/data"
    # # store_data_dir = "/home/data_8T/EMPAIR/cryobench/Ribosembly/images"
    # store_data_dir = "/home/data_8T/EMPAIR/10841"
    # # store_data_dir = "/home/data_8T/EMPAIR/10059/data/particles"
    # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # # img_size = 320
    # # img_size = 256
    # # img_size = 128
    # img_size = 160
    # # img_size = 192
    # # img_apix = 1.699
    # # img_apix = 1.2156
    # # img_apix = 1.31
    # # img_apix = 3.0
    # img_apix = 2.64
    # # rots, trans = parse_pose_star(metafile, img_size, img_apix)
    # # # rots, trans = parse_pose_csparc(metafile)
    # # # print(rots.shape)
    # # # print(trans.shape)
    # # # print(trans[0])
    # # rots = rots.reshape(-1, 9)
    # # # trans *= 128 / 320 ## convert to downsample size
    # # # orientations = np.concatenate([rots[:50000], trans[:50000]], axis=1)
    # # orientations = np.concatenate([rots, trans], axis=1)
    # # print(orientations.shape)
    # # # print(orientations[:5])
    # # orientation_dir = "%s/poses_relion.npy" % gaussian_dir
    # # # orientation_dir = "%s/poses_cs_128.npy" % gaussian_dir
    # # # orientation_dir = "%s/poses_cs_j5.npy" % gaussian_dir
    # # np.save(orientation_dir, orientations)

    # # pose_file = "%s/poses.pkl" % source_dir
    # pose_file = "%s/poses.pkl" % gaussian_dir
    # # pose_file = "%s/poses_128.pkl" % gaussian_dir
    # with open(pose_file, 'rb') as f:
    #     poses = pickle.load(f)

    # rots, trans = poses[0], poses[1]
    # trans *= img_size
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)
    # print(orientations.shape)
    # # print(orientations[0]) 
    # orientation_dir = "%s/poses_relion.npy" % gaussian_dir
    # # orientation_dir = "%s/poses_cs.npy" % gaussian_dir
    # # orientation_dir = "%s/poses_cs_128.npy" % gaussian_dir
    # np.save(orientation_dir, orientations)

    # # ctf_file = "%s/ctf.pkl" % source_dir
    # # ctf_file = "%s/ctf_cs.pkl" % source_dir
    # ctf_file = "%s/ctf.pkl" % gaussian_dir
    # # ctf_file = "%s/ctf_128.pkl" % gaussian_dir
    # with open(ctf_file, 'rb') as f:
    #     ctf = pickle.load(f)
    #     # print(ctf.shape)
    #     # print(ctf[:5])
    # # ctf[:, 0] = 256
    # # ctf[:, 1] = 320 * 1.699 / 256
    # # print(ctf[:5])
    # # print(np.mean(ctf, axis=0))
    # ctf_output = "%s/ctf_relion.npy" % gaussian_dir
    # # ctf_output = "%s/ctf_cs.npy" % gaussian_dir
    # # ctf_output = "%s/ctf_cs_128.npy" % gaussian_dir
    # np.save(ctf_output, ctf)
    # # ctf_setting = ctf_params(ctf[0])
    # # ctf, defocus_list = compute_full_ctf(img_size, 50000, ctf_setting)
    # # # print(ctf.shape)
    # # ctf_output = "%s/ctf.npy" % source_dir
    # # ctf = ctf.cpu().data.numpy()
    # # np.save(ctf_output, ctf)


    # src = ImageSource.from_file(metafile, lazy=True, indices=None, datadir=store_data_dir)
    # print(src.shape)

    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src.dtype,
    #     is_vol=False,
    # )

    # output_mrc = "%s/img_stack1.mrcs" % source_dir
    # src.write_mrc(
    #     output_file=output_mrc,
    #     header=header,
    #     transform_fn=None,
    #     chunksize=5000,
    # )

    # image_mrc = "%s/img_stack1.mrc" % store_data_dir
    # with mrcfile.mmap(image_mrc, "r+") as mrc:
    #     data = mrc.data
    #     # print(data.shape)
    # # print(np.max(data[1]), np.min(data[1]))
    # # data = data * -1 ##  dark <--> light
    # data_subset = data[:50000, :, :] * -1

    # image_subset_mrc = "%s/img_stack.mrcs" % source_dir
    # with mrcfile.new(image_subset_mrc) as mrc:
    #     mrc.set_data(data_subset)

    # img_size = 128
    # img_apix = img_apix * 320 / 128


    # # init_map = "%s/cryosparc_initial_map.mrc" % source_dir
    # init_map = "%s/cryosparc_P15_J24_volume_map.mrc" % source_dir
    # # init_map = "%s/cryosparc_P25_J6_005_volume_map.mrc" % source_dir
    # # init_map = "%s/emd_5778.map" % source_dir
    # # init_map = "%s/cryosparc_P17_J7_004_volume_map.mrc" % source_dir
    # # init_map = "%s/cryosparc_P17_J7_004_volume_map_128.mrc" % source_dir
    # # init_map = "%s/emd_24562_trans.map" % source_dir
    # # init_map = "%s/emd_24562_trans.map" % gaussian_dir
    # # init_map = "%s/cryosparc_P25_J4_class_00_final_volume.mrc" % source_dir
    # # init_map = "%s/cryosparc_P25_J5_005_volume_map.mrc" % source_dir
    # # init_map = "%s/emd_8117_trans.map" % source_dir
    # with mrcfile.open(init_map, 'r') as mrc:
    #     vol = mrc.data 
    # # print(np.mean(vol), np.std(vol))
    # density_thres = 0.345
    # # density_thres = 0.235
    # # density_thres = 0.163
    # # density_thres = 0.454
    # # density_thres = 3.54
    # # density_thres = 0.04
    # # density_thres = 1.29
    # # density_thres = 3.41
    # # density_thres = 5.17

    # # vol = spherical_mask(img_size, radius=img_size / 4).cpu().numpy()
    # # density_thres = 0.9

    # density_mask = vol > density_thres
    # n_points = 20000
    # offOrigin = np.array([0, 0, 0])
    # dVoxel = np.array([img_apix, img_apix, img_apix])
    # real_size = img_size * img_apix
    # sVoxel = np.array([real_size, real_size, real_size])
    # # interval = (1, 1, 1)
    # interval = (2, 2, 2)
    # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # print(sampled_indices.shape)
    # # valid_indices = np.argwhere(density_mask)
    # # print(valid_indices.shape)
    # # sampled_indices = valid_indices[
    # #     np.random.choice(len(valid_indices), n_points, replace=False)
    # # ]
    # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    # # downsampled vol needs mutiply
    # # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 / 2 + offOrigin
    # # sampled_positions *= 2  

    # # sampled_densities = vol[
    # #     sampled_indices[:, 0],
    # #     sampled_indices[:, 1],
    # #     sampled_indices[:, 2],
    # # ]

    # standardize_vol = (vol - vol.mean()) / vol.std()
    # sampled_densities = standardize_vol[
    #     sampled_indices[:, 0],
    #     sampled_indices[:, 1],
    #     sampled_indices[:, 2],
    # ]

    # # sampled_densities /= 10.1
    # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # # print(np.max(sampled_positions, axis=0), np.min(sampled_positions, axis=0))
    # # save_path = "%s/init_gaussians_1inter.npy" % gaussian_dir
    # # save_path = "%s/emd_gaussians_1inter.npy" % gaussian_dir
    # save_path = "%s/emd_gaussians_2inter.npy" % gaussian_dir
    # # save_path = "%s/gaussians_1inter_128.npy" % gaussian_dir
    # # save_path = "%s/gaussians_2inter_128.npy" % gaussian_dir
    # # save_path = "%s/spherical_2inter.npy" % gaussian_dir
    # np.save(save_path, out)


    # img_size = 256
    # img_apix = img_apix * 320 / 256

    # # cfg_file = "%s/cfg_scale_0.5_2.0.json" % gaussian_dir
    # # cfg_file = "%s/cfg_scale_0.25_0.75.json" % gaussian_dir
    # # cfg_file = "%s/cfg_scale_0.5_1.5.json" % gaussian_dir
    # cfg_file = "%s/cfg_scale_0.5_1.0.json" % gaussian_dir
    # # cfg_file = "%s/cfg_scale_128_0.25_0.75.json" % gaussian_dir
    # meta_data = {}
    # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # scanner_cfg = {}
    # scanner_cfg['mode'] = "parallel"
    # scanner_cfg['filter'] = None
    # scanner_cfg['DSD'] = 7.0
    # scanner_cfg['DSO'] = 5.0
    # real_size = img_size * img_apix
    # scanner_cfg['nDetector'] = [img_size, img_size]
    # scanner_cfg['sDetector'] = [real_size, real_size]
    # scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # scanner_cfg['offOrigin'] = [0, 0, 0]
    # scanner_cfg['offDetector'] = [0, 0, 0]
    # scanner_cfg['accuracy'] = 0.5
    # scanner_cfg['ctf'] = True
    # # scanner_cfg['invert_contrast'] = False
    # scanner_cfg['invert_contrast'] = True
    # scanner_cfg['pixelSize'] = img_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 2.0 / real_size

    # # scanner_cfg['scale_min'] = 0.25 / real_size
    # # scanner_cfg['scale_max'] = 0.75 / real_size

    # scanner_cfg['scale_min'] = 0.5 / real_size
    # scanner_cfg['scale_max'] = 1.0 / real_size

    # scanner_cfg['std'] = 0.7967
    # meta_data['cfg'] = scanner_cfg

    # with open(cfg_file, "w") as f:
    #     json.dump(meta_data, f)
    


    #### half split Relion
    # # source_dir = "/home/data_8T/EMPAIR/10180"
    # # metafile = "%s/consensus_data.star" % source_dir
    # # raw_data_dir = "%s/data" % source_dir
    # # store_data_dir = "%s/gaussian_preprocess" % source_dir
    # # img_size = 320
    # # img_apix = 1.699

    # source_dir = "/home/data_8T/EMPAIR/10005"
    # metafile = "%s/data/particles/tv1_relion_data.star" % source_dir
    # store_data_dir = "/home/data_8T/EMPAIR/10005/data/particles"
    # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # img_size = 256
    # img_apix = 1.2156

    # relionfile = Relionfile(metafile)
    # # print(relionfile.df['_rlnRandomSubset'].value_counts())

    # subset_indices1 = relionfile.df[relionfile.df['_rlnRandomSubset'] == '1'].index
    # # print(subset_indices1)
    # subset_indices2 = relionfile.df[relionfile.df['_rlnRandomSubset'] == '2'].index
    # # print(type(subset_indices1))

    # rots, trans = parse_pose_star(metafile, img_size, img_apix)
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)
    # orientation_dir1 = "%s/poses_relion1.npy" % gaussian_dir
    # orientation_dir2 = "%s/poses_relion2.npy" % gaussian_dir
    # np.save(orientation_dir1, orientations[subset_indices1])
    # np.save(orientation_dir2, orientations[subset_indices2])

    # ctf_file = "%s/ctf_relion.pkl" % source_dir
    # with open(ctf_file, 'rb') as f:
    #     ctf = pickle.load(f)
    # ctf_halfset1 = ctf[subset_indices1]
    # ctf_halfset2 = ctf[subset_indices2]
    # ctf_dir1 = "%s/ctf_relion1.npy" % gaussian_dir
    # ctf_dir2 = "%s/ctf_relion2.npy" % gaussian_dir
    # np.save(ctf_dir1, ctf_halfset1)
    # np.save(ctf_dir2, ctf_halfset2)

    # image_subset1 = "%s/img_stack1.mrcs" % gaussian_dir
    # # test_indice = np.array([0, 1, 2])
    # # src1 = ImageSource.from_file(metafile, lazy=False, indices=test_indice, datadir=raw_data_dir)
    # src1 = ImageSource.from_file(metafile, lazy=True, indices=subset_indices1, datadir=store_data_dir)
    # # print(src1.shape)
    # # print(src1.shape, type(src1.data))
    # # print(np.max(src1.data[1]), np.min(src1.data[1]))

    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src1.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src1.dtype,
    #     is_vol=False,
    # )
    # # src1.write_mrc(
    # #     output_file=image_subset1,
    # #     header=header,
    # #     transform_fn=sign_flip,
    # #     chunksize=5000,
    # # )
    # src1.write_mrc(
    #     output_file=image_subset1,
    #     header=header,
    #     transform_fn=None,
    #     chunksize=5000,
    # )
    
    # image_subset2 = "%s/img_stack2.mrcs" % gaussian_dir
    # src2 = ImageSource.from_file(metafile, lazy=True, indices=subset_indices2, datadir=store_data_dir)
    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src2.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src2.dtype,
    #     is_vol=False,
    # )
    # # src2.write_mrc(
    # #     output_file=image_subset2,
    # #     header=header,
    # #     transform_fn=sign_flip,
    # #     chunksize=5000,
    # # )
    # src2.write_mrc(
    #     output_file=image_subset2,
    #     header=header,
    #     transform_fn=None,
    #     chunksize=5000,
    # )


    ### half split CryoSPARC
    # source_dir = "/home/data_8T/EMPAIR/10180"
    # metafile = "%s/cryosparc_particles.cs" % source_dir
    # raw_data_dir = "%s/data" % source_dir
    # store_data_dir = "%s/gaussian_preprocess" % source_dir
    # img_size = 320
    # img_apix = 1.699

    # source_dir = "/home/data_8T/EMPAIR/10005"
    # metafile = "%s/cryosparc_P25_J6_005_particles.cs" % source_dir
    # store_data_dir = "/home/data_8T/EMPAIR/10005/data/particles"
    # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # img_size = 256
    # img_apix = 1.2156

    # csdata = np.load(metafile)
    # # print(relionfile.df['_rlnRandomSubset'].value_counts())
    # subset_indices1 = []
    # subset_indices2 = []
    # for i in range(len(csdata)):
    #     if csdata[i][24] == 0:
    #         subset_indices1.append(i)
    #     else:
    #         subset_indices2.append(i)
    # subset_indices1 = np.array(subset_indices1)
    # subset_indices2 = np.array(subset_indices2)
    # # print((len(subset_indices1)), len(subset_indices2))
    # # print(subset_indices1[:8], subset_indices1[-8:])

    # # ctf_file = "%s/ctf_cs.pkl" % source_dir
    # # with open(ctf_file, 'rb') as f:
    # #     ctf = pickle.load(f)
    # ctf_file = "%s/ctf_cs.npy" % gaussian_dir
    # ctf = np.load(ctf_file)
    # ctf_halfset1 = ctf[subset_indices1]
    # ctf_halfset2 = ctf[subset_indices2]
    # ctf_dir1 = "%s/ctf_cs1.npy" % gaussian_dir
    # ctf_dir2 = "%s/ctf_cs2.npy" % gaussian_dir
    # np.save(ctf_dir1, ctf_halfset1)
    # np.save(ctf_dir2, ctf_halfset2)

    # # rots, trans = parse_pose_csparc(metafile)
    # # rots = rots.reshape(-1, 9)
    # # orientations = np.concatenate([rots, trans], axis=1, dtype=np.float64)
    # # orientation_dir1 = "%s/poses1_cs.npy" % store_data_dir
    # # orientation_dir2 = "%s/poses2_cs.npy" % store_data_dir
    # # np.save(orientation_dir1, orientations[subset_indices1])
    # # np.save(orientation_dir2, orientations[subset_indices2])

    # poses_file = "%s/poses_cs.npy" % gaussian_dir
    # poses = np.load(poses_file)
    # poses_halfset1 = poses[subset_indices1]
    # poses_halfset2 = poses[subset_indices2]
    # poses_dir1 = "%s/poses_cs1.npy" % gaussian_dir
    # poses_dir2 = "%s/poses_cs2.npy" % gaussian_dir
    # np.save(poses_dir1, poses_halfset1)
    # np.save(poses_dir2, poses_halfset2)


    # image_subset1 = "%s/img_stack1_cs.mrcs" % store_data_dir
    # # test_indice = np.array([0, 10000, 100000, 300000])
    # src1 = ImageSource.from_file(metafile, lazy=False, indices=subset_indices1, datadir=store_data_dir)
    # # src1 = ImageSource.from_file(metafile, lazy=True, indices=None, datadir=raw_data_dir)
    # # print(src1.shape)
    # print(src1.shape, type(src1.data))
    # # print(np.max(src1.data[1]), np.min(src1.data[1]))

    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src1.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src1.dtype,
    #     is_vol=False,
    # )
    # src1.write_mrc(
    #     output_file=image_subset1,
    #     header=header,
    #     transform_fn=sign_flip,
    #     chunksize=5000,
    # )
    
    # image_subset2 = "%s/img_stack2_cs.mrcs" % store_data_dir
    # src2 = ImageSource.from_file(metafile, lazy=True, indices=subset_indices2, datadir=raw_data_dir)
    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src2.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src2.dtype,
    #     is_vol=False,
    # )
    # src2.write_mrc(
    #     output_file=image_subset2,
    #     header=header,
    #     transform_fn=sign_flip,
    #     chunksize=5000,
    # )


    ### modify cs file
    # # source_dir = "dataset/10180"
    # source_dir = "/home/data_8T/EMPAIR/10005"
    # # metafile = "%s/consensus_data.star" % source_dir
    # # metafile = "%s/cryosparc_P15_J24_particles.cs" % source_dir
    # metafile = "%s/cryosparc_P25_J6_005_particles.cs" % source_dir
    # data = np.load(metafile)

    # # # zero_count = 0
    # # # one_count = 1
    # # # for i in range(len(data)):
    # # #     if data[i][24] == 0:
    # # #         zero_count += 1
    # # #     else:
    # # #         one_count += 1
    # # # print(zero_count, one_count)

    # # print(data[0].dtype)
    # dt_list = []
    # for i in range(len(data.dtype)):
    #     # print(i, data.dtype[i], data[0][i])
    #     if i == 1:
    #         dt_list.append((data.dtype.names[i], 'S100'))
    #     else:
    #         dt_list.append((data.dtype.names[i], data.dtype[i]))
    # new_dt = np.dtype(dt_list)
    # # print(new_dt)

    # # # for i in range(len(new_data.dtype)):
    # # #     print(i, new_data.dtype.names[i], new_data[0][i])

    # # data = data.astype(new_dt)
    # # tag_list = ['20160622', '20160710', '20160716', '20160813', '20160820', '20160911']
    # # # for i in range(2):
    # # for i in range(len(data)):
    # #     temp_path = data[i][1].decode('utf-8')
    # #     # print("raw path: ", temp_path)
    # #     for tag in tag_list:
    # #         if tag in temp_path:
    # #             index = temp_path.find(tag)
    # #             corrected_path = "/home/data_8T/EMPAIR/10180/data/Micrographs_%s/%s" % (tag, temp_path[index:])
    # #             # corrected_path = "Micrographs_%s/%s" % (tag, temp_path[index:])
    # #     # print("corrected: ", corrected_path)
    # #     data[i][1] = corrected_path.encode('utf-8')
    # #     # print("updated path: ", data[i][1]) 
    # # new_metafile = "%s/cryosparc_particles.cs" % source_dir
    # # np.save(new_metafile, data)

    # data = data.astype(new_dt)
    # # for i in range(2):
    # for i in range(len(data)):
    #     temp_path = data[i][1].decode('utf-8')
    #     # print("raw path: ", temp_path)
    #     corrected_path = "10005/data/particles/tv1.mrcs"
    #     # print("corrected: ", corrected_path)
    #     data[i][1] = corrected_path.encode('utf-8')
    #     # print("updated path: ", data[i][1]) 
    # new_metafile = "%s/cryosparc_particles.cs" % source_dir
    # np.save(new_metafile, data)


    ### cryobench ribosome
    # ribosome_distribution = np.array([9076, 14378, 23547, 44366, 30647, 38500, 3915, 3980, 12740, 11975, 17988, 5001, 35367, 37448, 40540, 5772])
    # selected_vols = [2, 3, 5, 6, 7, 14]
    # distribution_preSum = []
    # k = 0
    # for i in range(len(ribosome_distribution)):
    #     k += ribosome_distribution[i]
    #     distribution_preSum.append(k)
    # # print(distribution_preSum)
    # selected_indices = []
    # for i in selected_vols:
    #     # single_vol_indices = [j for j in range(distribution_preSum[i-1], distribution_preSum[i])]
    #     if ribosome_distribution[i] < 10000:
    #         single_vol_indices = [j for j in range(distribution_preSum[i-1], distribution_preSum[i])]
    #     else:
    #         single_vol_indices = [j for j in range(distribution_preSum[i-1], distribution_preSum[i-1] + 10000)]
    #     selected_indices.append(single_vol_indices)
    
    # selected_indices = np.concatenate(selected_indices, axis=0)
    # print(selected_indices.shape)

    # indice_dir = "/home/data_8T/EMPAIR/cryobench/Ribosembly/selected_indices.npy"
    # np.save(indice_dir, selected_indices)


    # source_dir = "/home/data_8T/EMPAIR/cryobench/Ribosembly"
    # indice_dir = "%s/selected_indices.npy" % source_dir
    # store_data_dir = "%s/images" % source_dir
    # metafile = "%s/snr0.01.star" % store_data_dir
    # ctf_dir = "%s/combined_ctfs.pkl" % source_dir
    # pose_dir = "%s/combined_poses.pkl" % source_dir
    # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # img_size = 128
    # img_apix = 3.0

    # selected_indices = np.load(indice_dir)
    # with open(ctf_dir, 'rb') as f:
    #     ctf = pickle.load(f)
    
    # selected_ctf = ctf[selected_indices]
    # # print(selected_ctf.shape)

    # with open(pose_dir, 'rb') as f:
    #     poses = pickle.load(f)

    # # print(poses[0].shape, poses[1].shape)
    # # print(poses[1][:5])
    # rots, trans = poses[0][selected_indices], poses[1][selected_indices]
    # trans *= img_size
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)

    # save_ctf = "%s/ctf.npy" % gaussian_dir
    # save_pose = "%s/poses.npy" % gaussian_dir

    # print("ctf shape: ", selected_ctf.shape)
    # np.save(save_ctf, selected_ctf)
    # print("poses shape: ", orientations.shape)
    # np.save(save_pose, orientations)
    
    # image_subset1 = "%s/img_stack.mrcs" % gaussian_dir
    # src1 = ImageSource.from_file(metafile, lazy=True, indices=selected_indices, datadir=store_data_dir)
    # # print(src1.shape)
    # # print(src1.shape, type(src1.data))
    # # print(np.max(src1.data[1]), np.min(src1.data[1]))

    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src1.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src1.dtype,
    #     is_vol=False,
    # )
    # src1.write_mrc(
    #     output_file=image_subset1,
    #     header=header,
    #     transform_fn=None,
    #     chunksize=5000,
    # )


    # init_map = "%s/sample_vol.mrc" % source_dir
    # # init_map = "%s/cryosparc_P25_J5_005_volume_map.mrc" % source_dir
    # with mrcfile.open(init_map, 'r') as mrc:
    #     vol = mrc.data 
    # density_thres = 0.0126
    # density_mask = vol > density_thres
    # offOrigin = np.array([0, 0, 0])
    # dVoxel = np.array([img_apix, img_apix, img_apix])
    # real_size = img_size * img_apix
    # sVoxel = np.array([real_size, real_size, real_size])
    # interval = (1, 1, 1)
    # # interval = (2, 2, 2)
    # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # print(sampled_indices.shape)
    # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    # sampled_densities = vol[
    #     sampled_indices[:, 0],
    #     sampled_indices[:, 1],
    #     sampled_indices[:, 2],
    # ]

    # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # save_path = "%s/gaussians_1inter.npy" % gaussian_dir
    # np.save(save_path, out)


    # cfg_file = "%s/cfg_scale_0.25_0.75.json" % gaussian_dir
    # meta_data = {}
    # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # scanner_cfg = {}
    # scanner_cfg['mode'] = "parallel"
    # scanner_cfg['filter'] = None
    # scanner_cfg['DSD'] = 7.0
    # scanner_cfg['DSO'] = 5.0
    # real_size = img_size * img_apix
    # scanner_cfg['nDetector'] = [img_size, img_size]
    # scanner_cfg['sDetector'] = [real_size, real_size]
    # scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # scanner_cfg['offOrigin'] = [0, 0, 0]
    # scanner_cfg['offDetector'] = [0, 0, 0]
    # scanner_cfg['accuracy'] = 0.5
    # scanner_cfg['ctf'] = True
    # scanner_cfg['invert_contrast'] = True
    # scanner_cfg['pixelSize'] = img_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 2.0 / real_size

    # scanner_cfg['scale_min'] = 0.25 / real_size
    # scanner_cfg['scale_max'] = 0.75 / real_size
    # meta_data['cfg'] = scanner_cfg

    # with open(cfg_file, "w") as f:
    #     json.dump(meta_data, f)



    ### cryobench IgG-RL
    # source_dir = "/home/data_8T/EMPAIR/cryobench/IgG-RL"
    # store_data_dir = "%s/images/snr0.01" % source_dir
    # metafile = "%s/snr0.01.star" % store_data_dir
    # ctf_dir = "%s/combined_ctfs.pkl" % source_dir
    # pose_dir = "%s/combined_poses.pkl" % source_dir
    # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # img_size = 128
    # img_apix = 3.0

    # selected_indices = np.array([j for j in range(39000, 59000)])

    # with open(ctf_dir, 'rb') as f:
    #     ctf = pickle.load(f)

    # with open(pose_dir, 'rb') as f:
    #     poses = pickle.load(f)

    # selected_ctf = ctf[selected_indices]

    # rots, trans = poses[0][selected_indices], poses[1][selected_indices]
    # trans *= img_size
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)
    # save_ctf = "%s/ctf.npy" % gaussian_dir
    # save_pose = "%s/poses.npy" % gaussian_dir
    # np.save(save_ctf, selected_ctf)
    # np.save(save_pose, orientations)

    # image_subset1 = "%s/img_stack.mrcs" % gaussian_dir
    # src1 = ImageSource.from_file(metafile, lazy=True, indices=selected_indices, datadir=store_data_dir)
    # print(src1.shape)
    # # print(src1.shape, type(src1.data))
    # # print(np.max(src1.data[1]), np.min(src1.data[1]))

    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src1.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src1.dtype,
    #     is_vol=False,
    # )
    # src1.write_mrc(
    #     output_file=image_subset1,
    #     header=header,
    #     transform_fn=None,
    #     chunksize=5000,
    # )

    # init_map = "%s/sample_vol.mrc" % source_dir
    # with mrcfile.open(init_map, 'r') as mrc:
    #     vol = mrc.data 
    # density_thres = 0.00467
    # density_mask = vol > density_thres
    # offOrigin = np.array([0, 0, 0])
    # dVoxel = np.array([img_apix, img_apix, img_apix])
    # real_size = img_size * img_apix
    # sVoxel = np.array([real_size, real_size, real_size])
    # interval = (1, 1, 1)
    # # interval = (2, 2, 2)
    # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # print(sampled_indices.shape)
    # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    # sampled_densities = vol[
    #     sampled_indices[:, 0],
    #     sampled_indices[:, 1],
    #     sampled_indices[:, 2],
    # ]

    # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # save_path = "%s/gaussians_1inter.npy" % gaussian_dir
    # np.save(save_path, out)


    # # cfg_file = "%s/cfg_scale_0.5_1.5.json" % gaussian_dir
    # cfg_file = "%s/cfg_scale_0.25_0.75.json" % gaussian_dir
    # meta_data = {}
    # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # scanner_cfg = {}
    # scanner_cfg['mode'] = "parallel"
    # scanner_cfg['filter'] = None
    # scanner_cfg['DSD'] = 7.0
    # scanner_cfg['DSO'] = 5.0
    # real_size = img_size * img_apix
    # scanner_cfg['nDetector'] = [img_size, img_size]
    # scanner_cfg['sDetector'] = [real_size, real_size]
    # scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # scanner_cfg['offOrigin'] = [0, 0, 0]
    # scanner_cfg['offDetector'] = [0, 0, 0]
    # scanner_cfg['accuracy'] = 0.5
    # scanner_cfg['ctf'] = True
    # scanner_cfg['invert_contrast'] = True
    # scanner_cfg['pixelSize'] = img_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 1.5 / real_size

    # scanner_cfg['scale_min'] = 0.25 / real_size
    # scanner_cfg['scale_max'] = 0.75 / real_size
    # meta_data['cfg'] = scanner_cfg

    # with open(cfg_file, "w") as f:
    #     json.dump(meta_data, f)



    # ### cryobench IgG-1D
    # source_dir = "/home/data_8T/EMPAIR/cryobench/IgG-1D"
    # store_data_dir = "%s/images/snr0.01" % source_dir
    # metafile = "%s/snr0.01.star" % store_data_dir
    # ctf_dir = "%s/ctf.pkl" % source_dir
    # pose_dir = "%s/pose.pkl" % source_dir
    # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # # gaussian_dir = "%s/gaussian_preprocess_total" % source_dir
    # img_size = 128
    # img_apix = 3.0

    # # selected_indices = np.array([j for j in range(39000, 59000)])
    # selected_indices = np.array([j for j in range(10000, 89999)])

    # with open(ctf_dir, 'rb') as f:
    #     ctf = pickle.load(f)

    # with open(pose_dir, 'rb') as f:
    #     poses = pickle.load(f)

    # selected_ctf = ctf[selected_indices]

    # rots, trans = poses[0][selected_indices], poses[1][selected_indices]
    # # rots, trans = poses[0], poses[1]
    # trans *= img_size
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)
    # save_ctf = "%s/ctf_part.npy" % gaussian_dir
    # # save_ctf = "%s/ctf.npy" % gaussian_dir
    # save_pose = "%s/poses_part.npy" % gaussian_dir
    # # save_pose = "%s/poses.npy" % gaussian_dir
    # np.save(save_ctf, selected_ctf)
    # # np.save(save_ctf, ctf)
    # np.save(save_pose, orientations)

    # # image_subset1 = "%s/img_stack.mrcs" % gaussian_dir
    # image_subset1 = "%s/img_stack_part.mrcs" % gaussian_dir
    # src1 = ImageSource.from_file(metafile, lazy=True, indices=selected_indices, datadir=store_data_dir)
    # # src1 = ImageSource.from_file(metafile, lazy=True, datadir=store_data_dir)
    # print(src1.shape)
    # # print(src1.shape, type(src1.data))
    # # print(np.max(src1.data[1]), np.min(src1.data[1]))

    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src1.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src1.dtype,
    #     is_vol=False,
    # )
    # src1.write_mrc(
    #     output_file=image_subset1,
    #     header=header,
    #     transform_fn=None,
    #     chunksize=5000,
    # )


    # # init_map = "%s/sample_vol.mrc" % source_dir
    # init_map = "%s/vols/128_org/049.mrc" % source_dir
    # with mrcfile.open(init_map, 'r') as mrc:
    #     vol = mrc.data 
    # # density_thres = 0.0178
    # # density_thres = 0.000157
    # density_thres = 0.0228
    # density_mask = vol > density_thres
    # offOrigin = np.array([0, 0, 0])
    # dVoxel = np.array([img_apix, img_apix, img_apix])
    # real_size = img_size * img_apix
    # sVoxel = np.array([real_size, real_size, real_size])
    # interval = (1, 1, 1)
    # # interval = (2, 2, 2)
    # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # print(sampled_indices.shape)
    # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    # sampled_densities = vol[
    #     sampled_indices[:, 0],
    #     sampled_indices[:, 1],
    #     sampled_indices[:, 2],
    # ]

    # # sampled_densities = sampled_densities / sampled_densities.std()

    # # sampled_densities = vol[
    # #     sampled_indices[:, 0],
    # #     sampled_indices[:, 1],
    # #     sampled_indices[:, 2],
    # # ]

    # # print(np.max(sampled_densities))
    # # print(np.mean(sampled_densities))
    # # print(np.min(sampled_densities))

    # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # # save_path = "%s/gaussians_1inter_059.npy" % gaussian_dir
    # save_path = "%s/gaussians_1inter_049.npy" % gaussian_dir
    # np.save(save_path, out)


    # cfg_file = "%s/cfg_scale_0.5_1.0.json" % gaussian_dir
    # # cfg_file = "%s/cfg_scale_0.25_0.75.json" % gaussian_dir
    # # cfg_file = "%s/cfg_scale_0.15_0.35.json" % gaussian_dir
    # meta_data = {}
    # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # scanner_cfg = {}
    # scanner_cfg['mode'] = "parallel"
    # scanner_cfg['filter'] = None
    # scanner_cfg['DSD'] = 7.0
    # scanner_cfg['DSO'] = 5.0
    # real_size = img_size * img_apix
    # scanner_cfg['nDetector'] = [img_size, img_size]
    # scanner_cfg['sDetector'] = [real_size, real_size]
    # scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # scanner_cfg['offOrigin'] = [0, 0, 0]
    # scanner_cfg['offDetector'] = [0, 0, 0]
    # scanner_cfg['accuracy'] = 0.5
    # scanner_cfg['ctf'] = True
    # scanner_cfg['invert_contrast'] = True
    # scanner_cfg['pixelSize'] = img_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 1.0 / real_size

    # # scanner_cfg['scale_min'] = 0.25 / real_size
    # # scanner_cfg['scale_max'] = 0.75 / real_size

    # scanner_cfg['scale_min'] = 0.3 / img_size
    # scanner_cfg['scale_max'] = 0.6 / img_size

    # scanner_cfg['std'] = 8.7975

    # meta_data['cfg'] = scanner_cfg

    # with open(cfg_file, "w") as f:
    #     json.dump(meta_data, f)


    # ### 10076 sample
    # source_dir = "/home/data_8T/EMPAIR/10076"
    # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # img_size = 320
    # img_apix = 1.31

    # init_map = "%s/related_emd/emd_24562.map" % source_dir
    # with mrcfile.open(init_map, 'r') as mrc:
    #     vol = mrc.data 
    # density_thres = 5.1
    # density_mask = vol > density_thres
    # offOrigin = np.array([0, 0, 0])
    # dVoxel = np.array([img_apix, img_apix, img_apix])
    # real_size = img_size * img_apix
    # sVoxel = np.array([real_size, real_size, real_size])
    # # interval = (1, 1, 1)
    # interval = (2, 2, 2)
    # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # print(sampled_indices.shape)
    # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

    # sampled_densities = vol[
    #     sampled_indices[:, 0],
    #     sampled_indices[:, 1],
    #     sampled_indices[:, 2],
    # ]

    # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # save_path = "%s/emd_24562_2inter.npy" % gaussian_dir
    # np.save(save_path, out)



    ### 10841 select partial
    # source_dir = "/home/data_8T/EMPAIR/10841"
    # metafile = "%s/L17all.star" % source_dir
    # store_data_dir = "/home/data_8T/EMPAIR/10841"
    # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # img_size = 160
    # img_apix = 2.64
    # rots, trans = parse_pose_star(metafile, img_size, img_apix)
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)

    # classes = np.load("%s/class_num.npy" % gaussian_dir)
    # selected_indices = []
    # for i in range(len(classes)):
    #     if classes[i] in [1, 2, 3, 4, 5, 30, 31, 32, 24, 25, 26, 27, 29]:
    #         selected_indices.append(i)
    # selected_indices = np.array(selected_indices)

    # classes_partial = classes[selected_indices]
    # np.save("%s/class_num_partial.npy" % gaussian_dir, classes_partial)

    # print(selected_indices.shape)
    # orientation_dir = "%s/poses_relion_partial.npy" % gaussian_dir
    # orientations = orientations[selected_indices]
    # np.save(orientation_dir, orientations)

    # ctf_file = "%s/ctf_relion.pkl" % source_dir
    # with open(ctf_file, 'rb') as f:
    #     ctf = pickle.load(f)
    # ctf_output = "%s/ctf_relion_partial.npy" % gaussian_dir
    # ctf = ctf[selected_indices]
    # np.save(ctf_output, ctf)

    # src = ImageSource.from_file(metafile, lazy=True, indices=selected_indices, datadir=store_data_dir)
    # print(src.shape)

    # from mrcfile_em import MRCHeader
    # header = MRCHeader.make_default_header(
    #     nz=src.n,
    #     ny=img_size,
    #     nx=img_size,
    #     Apix=img_apix,
    #     dtype=src.dtype,
    #     is_vol=False,
    # )

    # output_mrc = "%s/L17partial.mrc" % source_dir
    # src.write_mrc(
    #     output_file=output_mrc,
    #     header=header,
    #     transform_fn=None,
    #     chunksize=5000,
    # )


    # # raw_star = "/home/data_8T/EMPAIR/cryobench/IgG-1D/simulated/snr0.01.star"
    # raw_star = "/home/data_8T/EMPAIR/simulation/snr0.01.star"
    # raw_data = Relionfile(raw_star)
    # data_df, data_optics = raw_data.df, raw_data.data_optics
    # # print(data_df.columns)
    # del data_df['_rlnImageName']
    # del data_df['_rlnImageDimensionality']
    # del data_df['_rlnOpticsGroup']
    # del data_df['_rlnRandomSubset']

    # # data_df['_rlnOriginX'] = data_df['_rlnOriginX'].astype(float) * 3.0
    # # data_df['_rlnOriginY'] = data_df['_rlnOriginY'].astype(float) * 3.0
    # # print(data_df['_rlnOriginX'].dtype)
    # data_df['_rlnImagePixelSize'] = 1.0
    # data_df['_rlnImageSize'] = 192
    # data_df = data_df.iloc[:3000]

    # # new_star = "/home/data_8T/EMPAIR/cryobench/IgG-1D/simulated/temp1.star"
    # new_star = "/home/data_8T/EMPAIR/simulation/temp.star"
    # from relion import write_star
    # write_star(new_star, data_df, data_optics)



    # ### 10180 partial filtered ind
    # source_dir = "/home/data_8T/EMPAIR/10180"
    # # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # # gaussian_dir = "%s/gaussian_preprocess_128" % source_dir
    # gaussian_dir = "%s/gaussian_preprocess_256" % source_dir
    # # gaussian_dir = "%s/gaussian_preprocess_filtered" % source_dir
    # img_size = 320
    # img_apix = 1.699
    # # img_down_size = 128
    # img_down_size = 256
    # img_down_apix = img_size * img_apix / img_down_size

    # # # ind_file = "%s/filtered.ind.pkl" % gaussian_dir
    # # # with open(ind_file, 'rb') as f:
    # # #     ind = pickle.load(f)
    # # #     print(ind.shape)

    # # ctf_file = "%s/ctf.pkl" % gaussian_dir
    # # with open(ctf_file, 'rb') as f:
    # #     ctf = pickle.load(f)
    # #     print(ctf.shape)
    # #     ctf[:, 0] = img_down_size
    # #     ctf[:, 1] = img_down_apix
    # #     print(ctf[0])

    # # pose_file = "%s/poses.pkl" % gaussian_dir
    # # with open(pose_file, 'rb') as f:
    # #     poses = pickle.load(f)

    # # selected_ctf = ctf
    # # # selected_ctf = ctf[ind]

    # # rots, trans = poses[0], poses[1]
    # # # rots, trans = poses[0][ind], poses[1][ind]
    # # print(trans)
    # # if img_down_size:
    # #     trans *= img_down_size
    # # else:
    # #     trans *= img_size
    # # rots = rots.reshape(-1, 9)
    # # orientations = np.concatenate([rots, trans], axis=1)
    # # # print(orientations)

    # # save_ctf = "%s/ctf.npy" % gaussian_dir
    # # save_pose = "%s/poses.npy" % gaussian_dir
    # # # save_ctf = "%s/ctf_filtered.npy" % gaussian_dir
    # # # save_pose = "%s/poses_filtered.npy" % gaussian_dir
    # # np.save(save_ctf, selected_ctf)
    # # np.save(save_pose, orientations)


    # # # init_map = "%s/cryosparc_P15_J24_volume_map.mrc" % source_dir
    # # init_map = "%s/emd_3683.map" % source_dir
    # # with mrcfile.open(init_map, 'r') as mrc:
    # #     vol = mrc.data 
    # # # density_thres = 0.345
    # # density_thres = 0.0242
    # # density_mask = vol > density_thres
    # # offOrigin = np.array([0, 0, 0])
    # # dVoxel = np.array([img_apix, img_apix, img_apix])
    # # real_size = img_size * img_apix
    # # sVoxel = np.array([real_size, real_size, real_size])
    # # # interval = (1, 1, 1)
    # # # interval = (2, 2, 2)
    # # interval = (3, 3, 3)
    # # # interval = (4, 4, 4)
    # # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # # print(sampled_indices.shape)
    # # # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
    # # sampled_positions = sampled_indices * 1.43 - (1.43 * 480) / 2 + offOrigin

    # # sampled_densities = vol[
    # #     sampled_indices[:, 0],
    # #     sampled_indices[:, 1],
    # #     sampled_indices[:, 2],
    # # ]

    # # # print(sampled_positions)

    # # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # # # save_path = "%s/gaussians_1inter_059.npy" % gaussian_dir
    # # # save_path = "%s/gaussians_2inter.npy" % gaussian_dir
    # # save_path = "%s/gaussians_3inter.npy" % gaussian_dir
    # # # save_path = "%s/gaussians_4inter.npy" % gaussian_dir
    # # np.save(save_path, out)


    # cfg_file = "%s/cfg_scale_0.5_1.0.json" % gaussian_dir
    # meta_data = {}
    # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # scanner_cfg = {}
    # scanner_cfg['mode'] = "parallel"
    # scanner_cfg['filter'] = None
    # scanner_cfg['DSD'] = 7.0
    # scanner_cfg['DSO'] = 5.0
    # real_size = img_size * img_apix
    # scanner_cfg['nDetector'] = [img_down_size, img_down_size]
    # scanner_cfg['sDetector'] = [real_size, real_size]
    # scanner_cfg['nVoxel'] = [img_down_size, img_down_size, img_down_size]
    # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # scanner_cfg['offOrigin'] = [0, 0, 0]
    # scanner_cfg['offDetector'] = [0, 0, 0]
    # scanner_cfg['accuracy'] = 0.5
    # scanner_cfg['ctf'] = True
    # scanner_cfg['invert_contrast'] = True
    # scanner_cfg['pixelSize'] = img_down_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 1.0 / real_size

    # # scanner_cfg['scale_min'] = 0.5 / ((128 * 3.0) ** 2 / real_size)
    # # scanner_cfg['scale_max'] = 1.0 / ((128 * 3.0) ** 2 / real_size)
    # # # scanner_cfg['scale_min'] = 1.0 / ((128 * 3.0) ** 2 / real_size)
    # # # scanner_cfg['scale_max'] = 1.5 / ((128 * 3.0) ** 2 / real_size)

    # # scanner_cfg['scale_min'] = 0.5 / img_down_size
    # # scanner_cfg['scale_max'] = 1.5 / img_down_size

    # scanner_cfg['scale_min'] = 1.0 / img_down_size
    # scanner_cfg['scale_max'] = 2.0 / img_down_size

    # scanner_cfg['std'] = 0.9775
    # # scanner_cfg['std'] = 2.0406
    # meta_data['cfg'] = scanner_cfg

    # with open(cfg_file, "w") as f:
    #     json.dump(meta_data, f)


    # ### 10059
    # source_dir = "/home/data_8T/EMPAIR/10059"
    # # gaussian_dir = "%s/gaussian_preprocess" % source_dir
    # gaussian_dir = "%s/gaussian_preprocess_128" % source_dir
    # img_size = 192
    # img_apix = 1.2156
    # img_down_size = 128
    # img_down_apix = img_size * img_apix / img_down_size


    # # ctf_file = "%s/ctf.pkl" % gaussian_dir
    # # with open(ctf_file, 'rb') as f:
    # #     ctf = pickle.load(f)
    # #     print(ctf[0])

    # # pose_file = "%s/poses.pkl" % gaussian_dir
    # # with open(pose_file, 'rb') as f:
    # #     poses = pickle.load(f)
    # #     print(poses[0].shape)

    # # selected_ctf = ctf
    # # rots, trans = poses[0], poses[1]
    # # trans *= img_size
    # # rots = rots.reshape(-1, 9)
    # # orientations = np.concatenate([rots, trans], axis=1)
    # # print(orientations)

    # ctf_file = "%s/ctf.pkl" % gaussian_dir
    # with open(ctf_file, 'rb') as f:
    #     ctf = pickle.load(f)
    #     print(ctf.shape)
    #     ctf[:, 0] = img_down_size
    #     ctf[:, 1] = img_down_apix
    #     print(ctf[0])

    # pose_file = "%s/poses.pkl" % gaussian_dir
    # with open(pose_file, 'rb') as f:
    #     poses = pickle.load(f)

    # selected_ctf = ctf
    # rots, trans = poses[0], poses[1]
    # print(trans * img_size * img_apix)
    # if img_down_size:
    #     trans *= img_down_size
    # else:
    #     trans *= img_size
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)
    # print(orientations)

    # save_ctf = "%s/ctf.npy" % gaussian_dir
    # save_pose = "%s/poses.npy" % gaussian_dir
    # np.save(save_ctf, selected_ctf)
    # np.save(save_pose, orientations)


    # # init_map = "%s/emd_8117_trans.map" % source_dir
    # init_map = "%s/cryosparc_P5_J22_007_volume_map.mrc" % source_dir
    # with mrcfile.open(init_map, 'r') as mrc:
    #     vol = mrc.data 
    # # density_thres = 3.5
    # density_thres = 0.239
    # density_mask = vol > density_thres
    # offOrigin = np.array([0, 0, 0])
    # dVoxel = np.array([img_apix, img_apix, img_apix])
    # real_size = img_size * img_apix
    # sVoxel = np.array([real_size, real_size, real_size])
    # interval = (2, 2, 2)
    # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # print(sampled_indices.shape)
    # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
    # # sampled_positions = sampled_indices * 1.43 - (1.43 * 480) / 2 + offOrigin

    # sampled_densities = vol[
    #     sampled_indices[:, 0],
    #     sampled_indices[:, 1],
    #     sampled_indices[:, 2],
    # ]

    # # print(sampled_positions)

    # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # save_path = "%s/gaussians_2inter.npy" % gaussian_dir
    # np.save(save_path, out)


    # # cfg_file = "%s/cfg_scale_0.5_1.0.json" % gaussian_dir
    # # meta_data = {}
    # # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # # scanner_cfg = {}
    # # scanner_cfg['mode'] = "parallel"
    # # scanner_cfg['filter'] = None
    # # scanner_cfg['DSD'] = 7.0
    # # scanner_cfg['DSO'] = 5.0
    # # real_size = img_size * img_apix
    # # scanner_cfg['nDetector'] = [img_size, img_size]
    # # scanner_cfg['sDetector'] = [real_size, real_size]
    # # scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    # # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # # scanner_cfg['offOrigin'] = [0, 0, 0]
    # # scanner_cfg['offDetector'] = [0, 0, 0]
    # # scanner_cfg['accuracy'] = 0.5
    # # scanner_cfg['ctf'] = True
    # # scanner_cfg['invert_contrast'] = True
    # # scanner_cfg['pixelSize'] = img_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 1.0 / real_size
    # # scanner_cfg['std'] = 0.7979
    # # meta_data['cfg'] = scanner_cfg

    # # with open(cfg_file, "w") as f:
    # #     json.dump(meta_data, f)


    # # cfg_file = "%s/cfg_scale_0.5_1.0.json" % gaussian_dir
    # # meta_data = {}
    # # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # # scanner_cfg = {}
    # # scanner_cfg['mode'] = "parallel"
    # # scanner_cfg['filter'] = None
    # # scanner_cfg['DSD'] = 7.0
    # # scanner_cfg['DSO'] = 5.0
    # # real_size = img_size * img_apix
    # # scanner_cfg['nDetector'] = [img_down_size, img_down_size]
    # # scanner_cfg['sDetector'] = [real_size, real_size]
    # # scanner_cfg['nVoxel'] = [img_down_size, img_down_size, img_down_size]
    # # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # # scanner_cfg['offOrigin'] = [0, 0, 0]
    # # scanner_cfg['offDetector'] = [0, 0, 0]
    # # scanner_cfg['accuracy'] = 0.5
    # # scanner_cfg['ctf'] = True
    # # scanner_cfg['invert_contrast'] = True
    # # scanner_cfg['pixelSize'] = img_down_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 1.0 / real_size

    # # # # scanner_cfg['scale_min'] = 0.5 / ((128 * 3.0) ** 2 / real_size)
    # # # # scanner_cfg['scale_max'] = 1.0 / ((128 * 3.0) ** 2 / real_size)
    # # # scanner_cfg['scale_min'] = 0.5 / ((128 * 3.0) ** 2 / real_size)
    # # # scanner_cfg['scale_max'] = 1.5 / ((128 * 3.0) ** 2 / real_size)
    # # scanner_cfg['std'] = 1.2658
    # # meta_data['cfg'] = scanner_cfg

    # # with open(cfg_file, "w") as f:
    # #     json.dump(meta_data, f)


    # ### 10345
    # source_dir = "/home/data_8T/EMPAIR/10345"
    # # gaussian_dir = "%s/gaussian_preprocess_128" % source_dir
    # gaussian_dir = "%s/gaussian_preprocess_256" % source_dir
    # img_size = 300
    # img_apix = 1.345
    # # img_down_size = 128
    # img_down_size = 256
    # img_down_apix = img_size * img_apix / img_down_size


    # # ctf_file = "%s/ctf.pkl" % gaussian_dir
    # # with open(ctf_file, 'rb') as f:
    # #     ctf = pickle.load(f)
    # #     print(ctf[0])

    # # pose_file = "%s/poses.pkl" % gaussian_dir
    # # with open(pose_file, 'rb') as f:
    # #     poses = pickle.load(f)
    # #     print(poses[0].shape)

    # # selected_ctf = ctf
    # # rots, trans = poses[0], poses[1]
    # # trans *= img_size
    # # rots = rots.reshape(-1, 9)
    # # orientations = np.concatenate([rots, trans], axis=1)
    # # print(orientations)


    # # ctf_file = "%s/ctf.pkl" % gaussian_dir
    # # with open(ctf_file, 'rb') as f:
    # #     ctf = pickle.load(f)
    # #     print(ctf.shape)
    # #     ctf[:, 0] = img_down_size
    # #     ctf[:, 1] = img_down_apix
    # #     print(ctf[0])

    # # pose_file = "%s/poses.pkl" % gaussian_dir
    # # with open(pose_file, 'rb') as f:
    # #     poses = pickle.load(f)

    # # selected_ctf = ctf
    # # rots, trans = poses[0], poses[1]
    # # print(trans * img_size * img_apix)
    # # if img_down_size:
    # #     trans *= img_down_size
    # # else:
    # #     trans *= img_size
    # # rots = rots.reshape(-1, 9)
    # # orientations = np.concatenate([rots, trans], axis=1)
    # # print(orientations)

    # # save_ctf = "%s/ctf.npy" % gaussian_dir
    # # save_pose = "%s/poses.npy" % gaussian_dir
    # # np.save(save_ctf, selected_ctf)
    # # np.save(save_pose, orientations)


    # # init_map = "%s/cryosparc_P4_J378_flex_map.mrc" % source_dir
    # # with mrcfile.open(init_map, 'r') as mrc:
    # #     vol = mrc.data 
    # # density_thres = 0.0195
    # # offOrigin = np.array([0, 0, 0])
    # # dVoxel = np.array([img_apix, img_apix, img_apix])
    # # real_size = img_size * img_apix
    # # sVoxel = np.array([real_size, real_size, real_size])
    # # interval = 1.75
    # # sampled_densities, sampled_indices = unifrom_sample_with_value_threshold(vol, interval, density_thres)
    # # # print(sampled_densities.shape)
    # # # print(sampled_indices.shape)

    # # sampled_positions = sampled_indices * (real_size / 256) - sVoxel / 2 + offOrigin

    # # # # print(sampled_positions)
    # # # low_density_regions = sampled_densities < 0.29
    # # # sampled_densities[low_density_regions] = sampled_densities[low_density_regions] + 0.1

    # # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # # save_path = "%s/gaussians_%.1finter.npy" % (gaussian_dir, interval)
    # # np.save(save_path, out)


    # # cfg_file = "%s/cfg_scale_0.5_1.0.json" % gaussian_dir
    # # meta_data = {}
    # # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # # scanner_cfg = {}
    # # scanner_cfg['mode'] = "parallel"
    # # scanner_cfg['filter'] = None
    # # scanner_cfg['DSD'] = 7.0
    # # scanner_cfg['DSO'] = 5.0
    # # real_size = img_size * img_apix
    # # scanner_cfg['nDetector'] = [img_size, img_size]
    # # scanner_cfg['sDetector'] = [real_size, real_size]
    # # scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    # # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # # scanner_cfg['offOrigin'] = [0, 0, 0]
    # # scanner_cfg['offDetector'] = [0, 0, 0]
    # # scanner_cfg['accuracy'] = 0.5
    # # scanner_cfg['ctf'] = True
    # # scanner_cfg['invert_contrast'] = True
    # # scanner_cfg['pixelSize'] = img_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 1.0 / real_size
    # # scanner_cfg['std'] = 0.7977
    # # meta_data['cfg'] = scanner_cfg

    # # with open(cfg_file, "w") as f:
    # #     json.dump(meta_data, f)


    # cfg_file = "%s/cfg_scale_0.5_1.0.json" % gaussian_dir
    # meta_data = {}
    # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # scanner_cfg = {}
    # scanner_cfg['mode'] = "parallel"
    # scanner_cfg['filter'] = None
    # scanner_cfg['DSD'] = 7.0
    # scanner_cfg['DSO'] = 5.0
    # real_size = img_size * img_apix
    # scanner_cfg['nDetector'] = [img_down_size, img_down_size]
    # scanner_cfg['sDetector'] = [real_size, real_size]
    # scanner_cfg['nVoxel'] = [img_down_size, img_down_size, img_down_size]
    # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # scanner_cfg['offOrigin'] = [0, 0, 0]
    # scanner_cfg['offDetector'] = [0, 0, 0]
    # scanner_cfg['accuracy'] = 0.5
    # scanner_cfg['ctf'] = True
    # scanner_cfg['invert_contrast'] = True
    # scanner_cfg['pixelSize'] = img_down_apix
    # # scanner_cfg['scale_min'] = 0.5 / real_size
    # # scanner_cfg['scale_max'] = 1.0 / real_size

    # # scanner_cfg['scale_min'] = 0.75 / real_size
    # # scanner_cfg['scale_max'] = 1.25 / real_size

    # scanner_cfg['scale_min'] = 0.75 / img_down_size
    # scanner_cfg['scale_max'] = 1.25 / img_down_size

    # # scanner_cfg['std'] = 2.0816  ### 128
    # scanner_cfg['std'] = 0.9705  ### 256
    # meta_data['cfg'] = scanner_cfg

    # with open(cfg_file, "w") as f:
    #     json.dump(meta_data, f)



    ### 10345
    source_dir = "/home/data_8T/EMPAIR/10841"
    gaussian_dir = "%s/gaussian_preprocess" % source_dir
    img_size = 160
    img_apix = 2.64

    # ctf_file = "%s/ctf.pkl" % gaussian_dir
    # with open(ctf_file, 'rb') as f:
    #     ctf = pickle.load(f)
    #     print(ctf[0])

    # pose_file = "%s/poses.pkl" % gaussian_dir
    # with open(pose_file, 'rb') as f:
    #     poses = pickle.load(f)
    #     print(poses[0].shape)

    # selected_ctf = ctf
    # rots, trans = poses[0], poses[1]
    # trans *= img_size
    # rots = rots.reshape(-1, 9)
    # orientations = np.concatenate([rots, trans], axis=1)
    # print(orientations)

    # save_ctf = "%s/ctf.npy" % gaussian_dir
    # save_pose = "%s/poses.npy" % gaussian_dir
    # np.save(save_ctf, selected_ctf)
    # np.save(save_pose, orientations)


    # init_map = "%s/emd_24562.map" % source_dir
    # with mrcfile.open(init_map, 'r') as mrc:
    #     vol = mrc.data 
    # density_thres = 5.1
    # density_mask = vol > density_thres
    # offOrigin = np.array([0, 0, 0])
    # dVoxel = np.array([img_apix, img_apix, img_apix])
    # real_size = img_size * img_apix
    # sVoxel = np.array([real_size, real_size, real_size])
    # # interval = (1, 1, 1)
    # interval = (3, 3, 3)
    # sampled_indices = grid_sample_with_value_threshold(vol, interval, density_thres, num_points=None)
    # print(sampled_indices.shape)
    # # sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
    # sampled_positions = sampled_indices * 1.31 - sVoxel / 2 + offOrigin

    # sampled_densities = vol[
    #     sampled_indices[:, 0],
    #     sampled_indices[:, 1],
    #     sampled_indices[:, 2],
    # ]

    # out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    # save_path = "%s/emd_24562_3inter.npy" % gaussian_dir
    # np.save(save_path, out)


    init_map = "%s/emd_24562.map" % source_dir
    flex_map = "%s/emd_24562_flex.mrc" % source_dir
    with mrcfile.open(init_map, 'r') as mrc:
        vol1 = mrc.data 
    with mrcfile.open(flex_map, 'r') as mrc:
        vol2 = mrc.data
    density_thres = 5.1
    offOrigin = np.array([0, 0, 0])
    dVoxel = np.array([img_apix, img_apix, img_apix])
    real_size = img_size * img_apix
    sVoxel = np.array([real_size, real_size, real_size])
    interval = (3, 3, 3)
    sampled_indices1 = grid_sample_with_value_threshold(vol1, interval, density_thres, num_points=None)
    print(sampled_indices1.shape)
    sampled_indices2 = grid_sample_with_value_threshold(vol2, interval, density_thres, num_points=None)
    print(sampled_indices2.shape)
    sampled_positions = sampled_indices2 * 1.31 - sVoxel / 2 + offOrigin

    sampled_densities = vol2[
        sampled_indices2[:, 0],
        sampled_indices2[:, 1],
        sampled_indices2[:, 2],
    ]

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    save_path = "%s/emd_24562_3inter_flex.npy" % gaussian_dir
    np.save(save_path, out)

    # mask_indices = []
    # for point in sampled_indices2:
    #     # 计算当前点与b中所有点的坐标差异（广播机制：(N,3) vs (3,) → (N,3)）
    #     diff = sampled_indices1 - point
    #     # 找到三维坐标完全匹配的位置（差异全为0）
    #     match_positions = np.where(np.all(diff == 0, axis=1))[0]
        
    #     # 处理匹配结果：有匹配则取第一个索引，无匹配则记为-1（可自定义）
    #     if len(match_positions) > 0:
    #         mask_indices.append(int(match_positions[0]))  # 转为int（避免NumPy类型）

    # total_indices = np.zeros(len(sampled_indices1))
    # mask_indices = np.array(mask_indices)
    # print(mask_indices.shape)
    # total_indices[mask_indices] = 1.0
    # save_path = "%s/mask_indices.npy" % gaussian_dir
    # np.save(save_path, total_indices)

    # cfg_file = "%s/cfg_scale_0.5_1.0.json" % gaussian_dir
    # meta_data = {}
    # meta_data['bbox'] = [[-1, -1, -1], [1, 1, 1]]
    # scanner_cfg = {}
    # scanner_cfg['mode'] = "parallel"
    # scanner_cfg['filter'] = None
    # scanner_cfg['DSD'] = 7.0
    # scanner_cfg['DSO'] = 5.0
    # real_size = img_size * img_apix
    # scanner_cfg['nDetector'] = [img_size, img_size]
    # scanner_cfg['sDetector'] = [real_size, real_size]
    # scanner_cfg['nVoxel'] = [img_size, img_size, img_size]
    # scanner_cfg['sVoxel'] = [2.0, 2.0, 2.0]
    # scanner_cfg['offOrigin'] = [0, 0, 0]
    # scanner_cfg['offDetector'] = [0, 0, 0]
    # scanner_cfg['accuracy'] = 0.5
    # scanner_cfg['ctf'] = True
    # scanner_cfg['invert_contrast'] = True
    # scanner_cfg['pixelSize'] = img_apix
    # scanner_cfg['scale_min'] = 0.5 / img_size
    # scanner_cfg['scale_max'] = 1.0 / img_size
    # scanner_cfg['std'] = 0.7962
    # meta_data['cfg'] = scanner_cfg

    # with open(cfg_file, "w") as f:
    #     json.dump(meta_data, f)
