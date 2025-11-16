import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from typing import Union, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt, binary_dilation
from utils_em import fftn_center, ifftn_center
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap


def spherical_window_mask(
    vol: Optional[Union[np.ndarray, torch.Tensor]] = None,
    *,
    D: Optional[int] = None,
    in_rad: float = 1.0,
    out_rad: float = 1.0,
) -> torch.Tensor:
    """Create a radial mask centered within a square image with a soft or hard edge.

    Given a volume or a volume's dimension, this function creates a masking array with
    values of 1.0 for points within `in_rad` of the image's center, values of 0.0 for
    points beyond `out_rad` of the center, and linearly-interpolated values between 0.0
    and 1.0 for points located between the two given radii.

    The default radii values create a mask circumscribed against the borders of the
    image with a hard edge.

    Arguments
    ---------
    vol:        A volume array to create a mask for.
    D:          Side length of the (square) image the mask is for.
    in_rad      Inner radius (fractional float between 0 and 1)
                inside which all values are 1.0
    out_rad     Outer radius (fractional float between 0 and 1)
                beyond which all values are 0.0

    Returns
    -------
    mask    A 2D torch.Tensor of shape (D,D) with mask values between
            0 (masked) and 1 (unmasked) inclusive.

    """
    if (vol is None) == (D is None):
        raise ValueError("Either `vol` or `D` must be specified!")
    if vol is not None:
        D = vol.shape[0]

    assert D % 2 == 0
    assert in_rad <= out_rad
    x0, x1 = torch.meshgrid(
        torch.linspace(-1, 1, D + 1, dtype=torch.float32)[:-1],
        torch.linspace(-1, 1, D + 1, dtype=torch.float32)[:-1],
        indexing="ij",
    )
    dists = (x0**2 + x1**2) ** 0.5

    # Create a mask with a hard edge which goes directly from 1.0 to 0.0
    if in_rad == out_rad:
        mask = (dists <= out_rad).float()

    # Create a mask with a soft edge between `in_rad` and `out_rad`
    else:
        mask = torch.minimum(
            torch.tensor(1.0),
            torch.maximum(torch.tensor(0.0), 1 - (dists - in_rad) / (out_rad - in_rad)),
        )

    return mask


def window_cos_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1., np.maximum(0.0, (r-in_rad)/(out_rad - in_rad)))
    mask = 0.5 + 0.5*np.cos(mask*np.pi)
    return mask



def cosine_dilation_mask(
    vol: Union[np.ndarray, torch.Tensor],
    threshold: Optional[float] = None,
    dilation: int = 25,
    edge_dist: int = 15,
    apix: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    threshold = threshold or np.percentile(vol, 99.99) / 2
    if verbose:
        print(f"Mask A/px={apix:.5g}; Threshold={threshold:.5g}")

    x = np.array(vol >= threshold).astype(bool)
    dilate_val = int(dilation // apix)
    if dilate_val:
        if verbose:
            print(f"Dilating initial vol>={threshold:3g} mask by {dilate_val} px")
        x = binary_dilation(x, iterations=dilate_val).astype(float)
    else:
        if verbose:
            print("no mask dilation applied")

    dist_val = edge_dist / apix
    if verbose:
        print(f"Width of cosine edge: {dist_val:.2f} px")

    if dist_val:
        y = distance_transform_edt(~x.astype(bool))
        y[y > dist_val] = dist_val
        z = np.cos(np.pi * y / dist_val / 2)
    else:
        z = x.astype(float)

    return z.round(6)


def get_fftn_center_dists(box_size: int) -> np.array:
    """Get distances from the center (and hence the resolution) for FFT co-ordinates."""

    x = np.arange(-box_size // 2, box_size // 2)
    x2, x1, x0 = np.meshgrid(x, x, x, indexing="ij")
    coords = np.stack((x0, x1, x2), -1)
    dists = (coords**2).sum(-1) ** 0.5
    assert dists[box_size // 2, box_size // 2, box_size // 2] == 0.0

    return dists


def calculate_fsc(
    v1: Union[np.ndarray, torch.Tensor], v2: Union[np.ndarray, torch.Tensor]
) -> float:
    """Calculate the Fourier Shell Correlation between two complex vectors."""
    var = (np.vdot(v1, v1) * np.vdot(v2, v2)) ** 0.5

    if var:
        fsc = float((np.vdot(v1, v2) / var).real)
    else:
        fsc = 1.0

    return fsc


def get_fsc_curve(
    vol1: torch.Tensor,
    vol2: torch.Tensor,
    initial_mask: Optional[torch.Tensor] = None,
    out_file: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate the FSCs between two volumes across all available resolutions."""

    # Apply the given mask before applying the Fourier transform
    maskvol1 = vol1 * initial_mask if initial_mask is not None else vol1.clone()
    maskvol2 = vol2 * initial_mask if initial_mask is not None else vol2.clone()
    box_size = vol1.shape[0]
    dists = get_fftn_center_dists(box_size)
    maskvol1 = fftn_center(maskvol1)
    maskvol2 = fftn_center(maskvol2)

    prev_mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    fsc = [1.0]
    for i in range(1, box_size // 2):
        mask = dists < i
        shell = np.where(mask & np.logical_not(prev_mask))
        fsc.append(calculate_fsc(maskvol1[shell], maskvol2[shell]))
        prev_mask = mask

    fsc_vals = pd.DataFrame(
        dict(pixres=np.arange(box_size // 2) / box_size, fsc=fsc), dtype=float
    )

    return fsc_vals


def get_fsc_thresholds(
    fsc_vals: pd.DataFrame, apix: float, verbose: bool = True
) -> Tuple[float, float]:
    """Retrieve the max resolutions at which an FSC curve is above 0.5 and 0.143."""

    if ((fsc_vals.pixres > 0) & (fsc_vals.fsc >= 0.5)).any():
        res_05 = fsc_vals.pixres[fsc_vals.fsc >= 0.5].max()
        if verbose:
            print("res @ FSC=0.5: {:.4g} ang".format((1 / res_05) * apix))
    else:
        res_05 = None
        if verbose:
            print("res @ FSC=0.5: N/A")

    if ((fsc_vals.pixres > 0) & (fsc_vals.fsc >= 0.143)).any():
        res_143 = fsc_vals.pixres[fsc_vals.fsc >= 0.143].max()
        if verbose:
            print("res @ FSC=0.143: {:.4g} ang".format((1 / res_143) * apix))
    else:
        res_143 = None
        if verbose:
            print("res @ FSC=0.143: N/A")

    return res_05, res_143


def randomize_phase(cval: complex) -> complex:
    """Create a new complex value with the same amplitude but scrambled phase."""
    amp = (cval.real**2.0 + cval.imag**2.0) ** 0.5
    angrand = np.random.random() * 2 * np.pi

    return complex(amp * np.cos(angrand), amp * np.sin(angrand))


def correct_fsc(
    fsc_vals: pd.DataFrame,
    vol1: torch.Tensor,
    vol2: torch.Tensor,
    randomization_threshold: float,
    initial_mask: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    """Apply phase-randomization null correction to given FSC volumes past a resolution.

    This function implements cryoSPARC-style correction to an FSC curve to account for
    the boost in FSCs that can be attributed to a mask that is too sharp or too tightly
    fits the volumes and thus introduces an artificial source of correlation.

    """
    box_size = vol1.shape[0]
    if fsc_vals.shape[0] != (box_size // 2):
        raise ValueError(
            f"Given FSC values must have (D // 2) + 1 = {(box_size // 2) + 1} entries, "
            f"instead have {fsc_vals.shape[0]}!"
        )

    # Randomize phases in the raw half-maps beyond the given threshold
    dists = get_fftn_center_dists(box_size)
    fftvol1 = fftn_center(vol1)
    fftvol2 = fftn_center(vol2)
    phase_res = int(randomization_threshold * box_size)
    rand_shell = np.where(dists >= phase_res)
    fftvol1[rand_shell] = fftvol1[rand_shell].apply_(randomize_phase)
    fftvol2[rand_shell] = fftvol2[rand_shell].apply_(randomize_phase)
    fftvol1 = ifftn_center(fftvol1)
    fftvol2 = ifftn_center(fftvol2)

    # Apply the given masks then go back into Fourier space
    maskvol1 = fftvol1 * initial_mask if initial_mask is not None else fftvol1.clone()
    maskvol2 = fftvol2 * initial_mask if initial_mask is not None else fftvol2.clone()
    maskvol1 = fftn_center(maskvol1)
    maskvol2 = fftn_center(maskvol2)

    # re-calculate the FSCs past the resolution using the phase-randomized volumes
    prev_mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    fsc = fsc_vals.fsc.tolist()
    for i in range(1, box_size // 2):
        mask = dists < i
        shell = np.where(mask & np.logical_not(prev_mask))

        if i > phase_res:
            p = calculate_fsc(maskvol1[shell], maskvol2[shell])

            # normalize the original FSC value using the phase-randomized value
            if p == 1.0:
                fsc[i] = 0.0
            elif not np.isnan(p):
                fsc[i] = np.clip((fsc[i] - p) / (1 - p), 0, 1.0)

        prev_mask = mask

    return pd.DataFrame(
        dict(pixres=np.arange(box_size // 2) / box_size, fsc=fsc), dtype=float
    )


def calculate_cryosparc_fscs(
    full_vol: torch.Tensor,
    half_vol1: torch.Tensor,
    half_vol2: torch.Tensor,
    sphere_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    loose_mask: Tuple[int, int] = (25, 15),
    tight_mask: Union[Tuple[int, int], np.ndarray] = (6, 6),
    apix: float = 1.0,
    out_file: Optional[str] = None,
    plot_file: Optional[str] = None,
) -> pd.DataFrame:
    """Calculating cryoSPARC-style FSC curves with phase randomization correction."""
    if sphere_mask is None:
        sphere_mask = spherical_window_mask(D=full_vol.shape[0])

    masks = {
        "No Mask": None,
        "Spherical": sphere_mask,
        "Loose": cosine_dilation_mask(
            full_vol, dilation=loose_mask[0], edge_dist=loose_mask[1], apix=apix
        ),
    }
    if isinstance(tight_mask, tuple):
        masks["Tight"] = cosine_dilation_mask(
            full_vol, dilation=tight_mask[0], edge_dist=tight_mask[1], apix=apix
        )
    elif isinstance(tight_mask, (np.ndarray, torch.Tensor)):
        masks["Tight"] = tight_mask
    else:
        raise TypeError(
            f"`tight_mask` must be an array or a tuple giving dilation and cosine edge "
            f"size in pixels, instead given {type(tight_mask).__name__}!"
        )

    fsc_vals = {
        mask_lbl: get_fsc_curve(half_vol1, half_vol2, initial_mask=mask)
        for mask_lbl, mask in masks.items()
    }
    fsc_thresh = {
        mask_lbl: get_fsc_thresholds(fsc_df, apix, verbose=False)[1]
        for mask_lbl, fsc_df in fsc_vals.items()
    }

    if fsc_thresh["Tight"] is not None:
        if fsc_thresh["Tight"] == fsc_vals["Tight"].pixres.values[-1]:
            rand_thresh = 0.5 * fsc_thresh["No Mask"]
        else:
            rand_thresh = 0.75 * fsc_thresh["Tight"]

        fsc_vals["Corrected"] = correct_fsc(
            fsc_vals["Tight"],
            half_vol1,
            half_vol2,
            randomization_threshold=rand_thresh,
            initial_mask=masks["Tight"],
        )
        fsc_thresh["Corrected"] = get_fsc_thresholds(
            fsc_vals["Corrected"], apix, verbose=False
        )[1]
    else:
        fsc_vals["Corrected"] = fsc_vals["Tight"]
        fsc_thresh["Corrected"] = fsc_thresh["Tight"]

    # Report corrected FSCs by printing FSC=0.5 and FSC=0.143 threshold values to screen
    get_fsc_thresholds(fsc_vals["Corrected"], apix)

    if plot_file is not None:
        fsc_angs = {
            mask_lbl: ((1 / fsc_val) * apix) for mask_lbl, fsc_val in fsc_thresh.items()
        }
        fsc_plot_vals = {
            f"{mask_lbl}  ({fsc_angs[mask_lbl]:.2f}Å)": fsc_df
            for mask_lbl, fsc_df in fsc_vals.items()
        }
        create_fsc_plot(fsc_vals=fsc_plot_vals, outfile=plot_file, apix=apix)

    pixres_index = {tuple(vals.pixres.values) for vals in fsc_vals.values()}
    assert len(pixres_index) == 1
    pixres_index = tuple(pixres_index)[0]

    fsc_vals = pd.DataFrame(
        {k: vals.fsc.values for k, vals in fsc_vals.items()}, index=list(pixres_index)
    )
    fsc_vals.index.name = "pixres"

    return fsc_vals


def plot_fsc_vals(fsc_arr: pd.DataFrame, label: str, **plot_args) -> None:
    """Add this set of FSC curves to the current plot using the given aesthetics."""
    plotting_args = dict(linewidth=3.1, alpha=0.81)
    plotting_args.update(**plot_args)

    # if one of the columns is the pixel resolution, use that as the x-axis...
    if "pixres" in fsc_arr.columns:
        for col in set(fsc_arr.columns) - {"pixres"}:
            plt.plot(fsc_arr.pixres, fsc_arr[col], label=col, **plotting_args)

    # ...otherwise just plot the values sequentially
    else:
        for col in set(fsc_arr.columns):
            plt.plot(fsc_arr[col], label=col, **plotting_args)


def create_fsc_plot(
    fsc_vals: Union[np.ndarray, pd.DataFrame, Dict[str, pd.DataFrame]],
    outfile: Optional[str] = None,
    apix: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    """Plot a given set of Fourier shell correlation values on a single canvas.

    Arguments
    ---------
    fsc_vals:   An array or DataFrame of FSC values, in which case each column will be
                treated as an FSC curve, or a dictionary of FSC curves expressed as
                DataFrames with an optional `pixres` columns.
    outfile:    Where to save the plot. If not given, plot will be displayed on screen.
    apix:       Supply an A/px value for creating proper x-axis frequency labels.
    title:      Optionally add this title to the plot.

    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if isinstance(fsc_vals, dict):
        for plt_lbl, fsc_array in fsc_vals.items():
            plot_fsc_vals(fsc_array, plt_lbl, linewidth=0.9 + 3.5 / len(fsc_vals))

    elif isinstance(fsc_vals, (np.ndarray, pd.DataFrame)):
        plot_fsc_vals(fsc_vals, "")

    else:
        raise TypeError(f"Unrecognized type for `fsc_vals`: {type(fsc_vals).__name__}!")

    # res_given = isinstance(fsc_vals, pd.DataFrame) and fsc_vals.shape[1] == 2
    res_given = isinstance(fsc_vals, pd.DataFrame)
    if isinstance(fsc_vals, dict):
        res_given |= all(fsc_arr.shape[1] == 2 for fsc_arr in fsc_vals.values())

    if res_given:
        use_xticks = np.arange(0.1, 0.6, 0.1)
        xtick_lbls = [f"1/{val:.1f}Å" for val in ((1 / use_xticks) * apix)]
        plt.xticks(use_xticks, xtick_lbls)
    elif apix is not None:
        print(
            f"Supplied A/px={apix} but can't produce frequency x-axis labels if "
            f"input arrays don't have `pixres` columns!"
        )

    # titles for axes
    plt.xlabel("Spatial frequency", size=14)
    plt.ylabel("Fourier shell correlation", size=14)

    plt.axhline(y=0.143, color="b", linewidth=1.4)
    plt.grid(True, linewidth=0.7, color="0.5", alpha=0.33)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.ylim(0, 1.0)
    plt.xlim(0, 0.5)
    ax.set_aspect(0.3)  # Set the aspect ratio on the plot specifically
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    if title:
        plt.title(title)

    # # Create the legend on the figure, not the plot
    # if isinstance(fsc_vals, dict):
    #     plt.legend(loc="best", prop={"size": 12})
    plt.legend()

    plt.show()



# Dimensionality reduction

def run_pca(z: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(z.shape[1])
    pca.fit(z)
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    pc = pca.transform(z)
    return pc, pca


def get_pc_traj(
    pca: PCA,
    zdim: int,
    numpoints: int,
    dim: int,
    start: Optional[float],
    end: Optional[float],
    percentiles: Optional[np.ndarray] = None,
) -> npt.NDArray[np.float32]:
    """
    Create trajectory along specified principal component

    Inputs:
        pca: sklearn PCA object from run_pca
        zdim (int)
        numpoints (int): number of points between @start and @end
        dim (int): PC dimension for the trajectory (1-based index)
        start (float): Value of PC{dim} to start trajectory
        end (float): Value of PC{dim} to stop trajectory
        percentiles (np.array or None): Define percentile array instead of np.linspace(start,stop,numpoints)

    Returns:
        np.array (numpoints x zdim) of z values along PC
    """
    if percentiles is not None:
        assert len(percentiles) == numpoints
    traj_pca = np.zeros((numpoints, zdim))
    if percentiles is not None:
        traj_pca[:, dim - 1] = percentiles
    else:
        assert start is not None
        assert end is not None
        traj_pca[:, dim - 1] = np.linspace(start, end, numpoints)
    ztraj_pca = pca.inverse_transform(traj_pca)
    return ztraj_pca


def run_tsne(
    z: np.ndarray, n_components: int = 2, perplexity: float = 1000
) -> np.ndarray:
    if len(z) > 10000:
        print("WARNING: {} datapoints > {}. This may take awhile.".format(len(z), 10000))
    z_embedded = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(z)
    return z_embedded


def run_umap(z: np.ndarray, **kwargs) -> np.ndarray:
    import umap  # CAN GET STUCK IN INFINITE IMPORT LOOP

    reducer = umap.UMAP(**kwargs)
    z_embedded = reducer.fit_transform(z)
    return z_embedded


# Clustering


def cluster_kmeans(
    z: np.ndarray, K: int, on_data: bool = True, reorder: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster z by K means clustering
    Returns cluster labels, cluster centers
    If reorder=True, reorders clusters according to agglomerative clustering of cluster centers
    """
    kmeans = KMeans(n_clusters=K, random_state=0, max_iter=10)
    labels = kmeans.fit_predict(z)
    centers = kmeans.cluster_centers_

    centers_ind = None
    if on_data:
        centers, centers_ind = get_nearest_point(z, centers)

    if reorder:
        g = sns.clustermap(centers)
        reordered = g.dendrogram_row.reordered_ind
        centers = centers[reordered]
        if centers_ind is not None:
            centers_ind = centers_ind[reordered]
        tmp = {k: i for i, k in enumerate(reordered)}
        labels = np.array([tmp[k] for k in labels])
    return labels, centers


def get_nearest_point(
    data: np.ndarray, query: np.ndarray
) -> Tuple[npt.NDArray[np.float32], np.ndarray]:
    """
    Find closest point in @data to @query
    Return datapoint, index
    """
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind


def choose_cmap(M):
    if M <= 10:
        cmap = "tab10"
    elif M <= 20:
        cmap = "tab20"
    else:
        cmap = ListedColormap(sns.color_palette("husl").as_hex())
    return cmap


def get_colors_for_cmap(cmap, M):
    if M <= 20:
        colors = plt.cm.get_cmap(cmap)(np.arange(M) / (np.ceil(M / 10) * 10))
    else:
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, M))
    return colors


