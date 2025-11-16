import numpy as np
import torch
from math import pi
from torch.fft import fftshift, ifftshift, fft2, fftn, ifftn

rad2deg = 180.0/pi
deg2rad = pi/180.0

def circular_mask(image_size, center=None, radius=None, bool=False):
    x = torch.arange(0, image_size).view(1, -1).repeat(image_size, 1)
    y = torch.arange(0, image_size).view(-1, 1).repeat(1, image_size)
    if center is None:
        center = (image_size / 2, image_size / 2)
    if radius is None:
        radius = image_size / 2
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    if bool:
        mask = (distance <= radius)
    else:
        mask = (distance <= radius).float()
    return mask


def spherical_mask(image_size, center=None, radius=None, bool=False):
    x, y, z = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), torch.arange(image_size), indexing="ij")
    if center is None:
        center = (image_size / 2, image_size / 2, image_size / 2)
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    if radius is None:
        radius = image_size / 2
    if bool:
        mask = (distance <= radius)
    else:
        mask = (distance <= radius).float()
    return mask

def relion_angle_to_matrix(angles):
    rads = [angle * deg2rad for angle in angles]
    R = np.zeros((3, 3))
    ca = np.cos(rads[0])
    cb = np.cos(rads[1])
    cg = np.cos(rads[2])
    sa = np.sin(rads[0])
    sb = np.sin(rads[1])
    sg = np.sin(rads[2])

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    R[0, 0] = cg * cc - sg * sa
    R[0, 1] = cg * cs + sg * ca
    R[0, 2] = -cg * sb
    R[1, 0] = -sg * cc - cg * sa
    R[1, 1] = -sg * cs + cg * ca
    R[1, 2] = sg * sb
    R[2, 0] = sc
    R[2, 1] = ss
    R[2, 2] = cb
    return R


def R_from_relion(eulers: np.ndarray):
    a = eulers[:, 0] * np.pi / 180.0
    b = eulers[:, 1] * np.pi / 180.0
    y = eulers[:, 2] * np.pi / 180.0
    nsamp = eulers.shape[0]
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    r_amat = np.array(
        [
            [ca, -sa, np.repeat(0, nsamp)],
            [sa, ca, np.repeat(0, nsamp)],
            [np.repeat(0, nsamp), np.repeat(0, nsamp), np.repeat(1, nsamp)],
        ],
    )
    r_bmat = np.array(
        [
            [cb, np.repeat(0, nsamp), -sb],
            [np.repeat(0, nsamp), np.repeat(1, nsamp), np.repeat(0, nsamp)],
            [sb, np.repeat(0, nsamp), cb],
        ]
    )
    r_ymat = np.array(
        [
            [cy, -sy, np.repeat(0, nsamp)],
            [sy, cy, np.repeat(0, nsamp)],
            [np.repeat(0, nsamp), np.repeat(0, nsamp), np.repeat(1, nsamp)],
        ]
    )
    rmat = np.matmul(np.matmul(r_ymat.T, r_bmat.T), r_amat.T)
    rmat[:, 0, 2] *= -1
    rmat[:, 2, 0] *= -1
    rmat[:, 1, 2] *= -1
    rmat[:, 2, 1] *= -1
    return rmat


def map_to_lie_algebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = v.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    R_y = v.new_tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    R_z = v.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    R = (
        R_x * v[..., 0, None, None]
        + R_y * v[..., 1, None, None]
        + R_z * v[..., 2, None, None]
    )
    return R


def expmap(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    K = map_to_lie_algebra(v / theta)

    I = torch.eye(3, device=v.device, dtype=v.dtype)  # noqa: E741
    R = (
        I
        + torch.sin(theta)[..., None] * K
        + (1.0 - torch.cos(theta))[..., None] * (K @ K)
    )
    return R

def fft2_center(img: torch.Tensor) -> torch.Tensor:
    """2-dimensional discrete Fourier transform reordered with origin at center."""
    if img.dtype == torch.float16:
        img = img.type(torch.float32)

    return fftshift(fft2(fftshift(img)))


def fftn_center(img: torch.Tensor) -> torch.Tensor:
    """N-dimensional discrete Fourier transform reordered with origin at center."""
    return fftshift(fftn(fftshift(img)))

def ifftn_center(img: torch.Tensor) -> torch.Tensor:
    """N-dimensional inverse discrete Fourier transform with origin at center."""
    return ifftshift(ifftn(ifftshift(img)))


def ht2_center(img: torch.Tensor) -> torch.Tensor:
    """2-dimensional discrete Hartley transform reordered with origin at center."""
    img = fft2_center(img)
    return img.real - img.imag


def htn_center(img: torch.Tensor) -> torch.Tensor:
    """N-dimensional discrete Hartley transform reordered with origin at center."""
    img = fftn_center(img)
    return img.real - img.imag


def iht2_center(img: torch.Tensor) -> torch.Tensor:
    """2-dimensional inverse discrete Hartley transform with origin at center."""
    img = fft2_center(img)
    img /= img.shape[-1] * img.shape[-2]
    return img.real - img.imag


def ihtn_center(img: torch.Tensor) -> torch.Tensor:
    """N-dimensional inverse discrete Hartley transform with origin at center."""
    img = fftn_center(img)
    img /= torch.prod(torch.tensor(img.shape, device=img.device))
    return img.real - img.imag


def symmetrize_ht(ht: torch.Tensor) -> torch.Tensor:
    if ht.ndim == 2:
        ht = ht[np.newaxis, ...]
    assert ht.ndim == 3
    n = ht.shape[0]

    D = ht.shape[-1]
    sym_ht = torch.empty((n, D + 1, D + 1), dtype=ht.dtype, device=ht.device)
    sym_ht[:, 0:-1, 0:-1] = ht

    assert D % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0, :]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]  # last corner is first corner

    if n == 1:
        sym_ht = sym_ht[0, ...]

    return sym_ht


if __name__ == "__main__":

    angle = [-82.77117, 92.442308, 120.011574]
    rot = relion_angle_to_matrix(angle)
    print(rot)

    euler = np.zeros((1, 3))
    euler[0, 0] = angle[0]
    euler[0, 1] = angle[1]
    euler[0, 2] = angle[2]
    rots = R_from_relion(euler)
    print(rots)
