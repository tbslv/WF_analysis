from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import (
    disk,
    binary_opening,
    binary_closing,
    binary_erosion,
)
from skimage import measure

from .dff_helper import _load_movie


# --------------------------------------------------
# Movie loading (priority matches DFF)
# --------------------------------------------------

def load_trial_movie_with_priority(
    *,
    out_dir: Path,
    raw_tiff_path: Path,
):
    """
    Load movie using the SAME priority logic as process_dff_trials().

    Priority:

    2) processed_root/.../<trial_id>/motion_corrected.npy
    3) processed_root/.../<trial_id>/motion_corrected.tiff
    4) raw_root/.../<trial_id>/*.tiff (raw_tiff_path)
    """

    mc_npy = out_dir / "motion_corrected.npy"
    mc_tiff = out_dir / "motion_corrected.tiff"

    if mc_npy.exists():
        return np.load(mc_npy), "motion_corrected"

    if mc_tiff.exists():
        return _load_movie(mc_tiff), "motion_corrected"

    return _load_movie(raw_tiff_path), "raw"


# --------------------------------------------------
# Mask estimation (YOUR implementation)
# --------------------------------------------------

def compute_brain_mask_simple(
    movie: np.ndarray,
    thr_val: float,
    offset: int = 0,
    smooth_radius: int = 2,
) -> np.ndarray:
    """
    Simple brain mask with smooth outline and inward offset.

    Parameters
    ----------
    movie : np.ndarray
        Raw movie (T, H, W)
    thr_val : float
        Threshold in normalized units (e.g. 0.1)
    offset : int
        Number of pixels to shrink the mask inward
    smooth_radius : int
        Radius (px) for outline smoothing

    Returns
    -------
    mask : np.ndarray (bool)
        Brain mask
    """

    # -------------------------
    # Mean + normalization
    # -------------------------
    mean_img = movie.mean(axis=0)

    vmin = float(np.min(mean_img))
    vmax = float(np.max(mean_img))

    if vmax > vmin:
        mean_norm = (mean_img - vmin) / (vmax - vmin)
    else:
        mean_norm = np.zeros_like(mean_img)

    # -------------------------
    # Threshold
    # -------------------------
    mask = mean_norm > thr_val

    # -------------------------
    # Fill + connectivity
    # -------------------------
    mask = binary_fill_holes(mask)

    lbl = label(mask)
    if lbl.max() > 1:
        areas = np.bincount(lbl.ravel())
        areas[0] = 0
        mask = lbl == areas.argmax()

    # -------------------------
    # Smooth outline
    # -------------------------
    if smooth_radius > 0:
        selem = disk(smooth_radius)
        mask = binary_opening(mask, selem)
        mask = binary_closing(mask, selem)

    # -------------------------
    # Inward offset (shrink)
    # -------------------------
    if offset > 0:
        selem = disk(offset)
        mask = binary_erosion(mask, selem)
        mask = enforce_border_offset(mask, 10)

    return mask

def enforce_border_offset(mask: np.ndarray, offset: int) -> np.ndarray:
    """
    Enforce a hard minimum distance from image borders.
    """
    if offset <= 0:
        return mask

    h, w = mask.shape
    mask[:offset, :] = False
    mask[-offset:, :] = False
    mask[:, :offset] = False
    mask[:, -offset:] = False

    return mask


def compute_mean_image(movie: np.ndarray) -> np.ndarray:
    """For diagnostics: mean image used as background."""
    return movie.mean(axis=0)


def apply_mask(movie: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a 2D mask to a 3D movie (T, H, W).
    Masked-out pixels are set to NaN.
    """
    masked = movie.copy()
    masked[:, ~mask] = np.nan
    return masked


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

def save_mask_diagnostic(
    mean_img: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
):
    """
    Save a diagnostic plot with mask contours overlaid.
    """
    contours = measure.find_contours(mask.astype(float), 0.5)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mean_img, cmap="gray")

    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color="red", linewidth=1)

    ax.set_title("Brain mask contour")
    ax.axis("off")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# High-level per-trial API
# --------------------------------------------------

def process_single_trial_mask(
    *,
    raw_tiff_path: Path,
    out_dir: Path,
    thr_val: float,
    offset: int = 0,
    smooth_radius: int = 2,
    overwrite: bool = True,
):
    """
    Full workflow for a single trial:
      - load movie with priority (masked > motion-corrected > raw)
      - compute brain mask via compute_brain_mask_simple()
      - apply mask to movie (NaN outside)
      - save masked.npy
      - save mask_contour.png

    Returns
    -------
    source : str
        "masked" | "motion_corrected" | "raw"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    out_masked_path = out_dir / "masked.npy"
    out_plot_path = out_dir / "mask_contour.png"

    if out_masked_path.exists() and not overwrite:
        return "masked"

    movie, source = load_trial_movie_with_priority(
        out_dir=out_dir,
        raw_tiff_path=raw_tiff_path,
    )

    mask = compute_brain_mask_simple(
        movie=movie,
        thr_val=thr_val,
        offset=int(offset),
        smooth_radius=int(smooth_radius),
    )

    masked_movie = apply_mask(movie, mask)
    np.save(out_masked_path, masked_movie)

    mean_img = compute_mean_image(movie)
    save_mask_diagnostic(mean_img, mask, out_plot_path)

    return source
