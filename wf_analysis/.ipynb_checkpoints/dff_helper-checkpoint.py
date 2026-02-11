
"""
DFF helper functions.

This version is self-contained:
- reads motion-corrected movies from processed_root (preferred) or raw_root
- computes ΔF/F movies
- saves dff.npy and mean.npy per trial

Expected folder layout for motion-corrected movies:
processed_root/<dataset>/<session>/<protocol>/<trial_id>/motion_corrected.npy  (or .tiff)

Raw TIFFs are still supported as a fallback if motion-corrected movies are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import tifffile as tiff
except Exception as e:
    raise ImportError("Please install 'tifffile' (pip install tifffile).") from e


def _load_movie(path: Path) -> np.ndarray:
    """Load either .npy or multi-page TIFF as (T,H,W) float32."""
    if path.suffix.lower() == ".npy":
        mov = np.load(path)
    else:
        mov = tiff.imread(str(path))
    if mov.ndim == 2:
        mov = mov[None, ...]
    if mov.ndim != 3:
        raise ValueError(f"Expected (T,H,W) movie; got shape={mov.shape} from {path}")
    return mov.astype(np.float32, copy=False)


def _downscale_2x(mov: np.ndarray) -> np.ndarray:
    """
    Simple 2x downscale by 2x2 binning. Requires even H/W; otherwise crops last row/col.
    """
    T, H, W = mov.shape
    H2 = (H // 2) * 2
    W2 = (W // 2) * 2
    mov = mov[:, :H2, :W2]
    mov = mov.reshape(T, H2 // 2, 2, W2 // 2, 2).mean(axis=(2, 4))
    return mov.astype(np.float32, copy=False)


def calc_dff_movie(
    mov: np.ndarray,
    baseline_frames: Tuple[int, int] = (0, 50),
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute ΔF/F for a movie.

    baseline_frames: (start, end) frames used for F0.
    """
    start, end = baseline_frames
    if end <= start or end > mov.shape[0]:
        raise ValueError(f"Invalid baseline_frames={baseline_frames} for movie length T={mov.shape[0]}")
    f0 = mov[start:end].mean(axis=0)
    f0 = np.maximum(f0, eps)
    dff = (mov - f0) / f0
    return dff.astype(np.float32, copy=False)


def process_dff_trials(
    dataset_id: str,
    session_id: str,
    raw_root: str | Path,
    processed_root: str | Path,
    protocol_list: List[str],
    file_name: str,
    baseline_start: int,
    baseline_end: int,
    response_start: int,
    response_end: int,
    do_downscale: bool,
    corrected_basename: str = "motion_corrected.npy",
) -> List[np.ndarray]:
    """
    Compute dff.npy per trial.

    Search order per trial:
    1) processed_root/.../<trial_id>/<corrected_basename> (default motion_corrected.npy)
       also accepts motion_corrected.tiff if corrected_basename is npy but tiff exists.
    2) raw_root/.../**/<file_name> as fallback

    Saves:
      processed_root/<dataset>/<session>/<protocol>/<trial_id>/dff.npy
      processed_root/<dataset>/<session>/<protocol>/<trial_id>/mean.npy  (mean dff in response window)
    """
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)

    dffs: List[np.ndarray] = []

    for protocol in protocol_list:
        raw_proto_dir = raw_root / dataset_id / session_id / protocol
        if not raw_proto_dir.exists():
            print(f"⚠️ Raw protocol folder not found: {raw_proto_dir}")
            continue

        raw_tiffs = sorted(raw_proto_dir.rglob(f"*{file_name}*"))
        if not raw_tiffs:
            print(f"⚠️ No raw TIFFs matching '*{file_name}*' under {raw_proto_dir}")
            continue

        print(f"\n=== DFF: {protocol} (N={len(raw_tiffs)}) ===")

        for raw_tiff in raw_tiffs:
            trial_id = raw_tiff.parent.name
            out_dir = processed_root / dataset_id / session_id / protocol / trial_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # Prefer masked to motion-corrected if present; fall back to raw tiff movie 
            masked = out_dir / "masked.npy"
            mc_npy = out_dir / corrected_basename
            mc_tiff = out_dir / "motion_corrected.tiff"
            
            if masked.exists():
                in_movie_path = masked
            elif mc_npy.exists():
                in_movie_path = mc_npy
            elif mc_tiff.exists():
                in_movie_path = mc_tiff
            else:
                in_movie_path = raw_tiff

            mov = _load_movie(in_movie_path)

            if do_downscale:
                mov = _downscale_2x(mov)

            dff = calc_dff_movie(
                mov,
                baseline_frames=(baseline_start, baseline_end),
            )

            mean_img = dff[response_start:response_end].mean(axis=0)

            np.save(out_dir / "dff.npy", dff)
            np.save(out_dir / "mean.npy", mean_img)

            dffs.append(dff)

    return dffs
