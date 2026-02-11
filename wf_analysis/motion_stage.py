"""
Motion correction stage for widefield TIFF recordings.

This stage discovers raw TIFF stacks under:
    raw_root/dataset_id/session_id/protocol/**/<file_name>

Then it runs motion correction using **only** the implementation in `wf_analysis/motion.py`
(the same functionality used in `dev_motion_correction.ipynb`).

Outputs per trial:
processed_root/<dataset>/<session>/<protocol>/<trial_id>/
    motion_corrected.npy
    motion_vectors.npy
    motion_reference.png
    motion_metadata.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

try:
    import tifffile as tiff
except Exception as e:  # pragma: no cover
    raise ImportError("Please install 'tifffile' (pip install tifffile).") from e

from .motion import (
    apply_motion_correction,
    save_motion_vectors,
    save_corrected_recording,
    save_reference_image,
    save_motion_correction_metadata,
)


@dataclass(frozen=True)
class MotionConfig:
    file_name: str = "recording.tiff"
    shift_method: str = "fourier"          # "integer" or "fourier"
    upsample_factor: int = 20               # subpixel precision (1/upsample_factor)
    n_parallel_workers: Optional[int] = 4
    motion_region: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None  # ((r0,c0),(r1,c1))
    corrected_basename: str = "motion_corrected.npy"


def _load_tiff_stack(path: Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D TIFF stack (t, y, x), got shape={arr.shape} at {path}")
    return arr


def _iter_trial_tiffs(raw_root: Path, dataset_id: str, session_id: str, protocol: str, file_name: str):
    # Matches the project convention of storing trials in nested folders.
    base = raw_root / dataset_id / session_id / protocol
    if not base.exists():
        return
    for p in base.rglob(file_name):
        yield p


def _parse_motion_region(value) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Accepts None, or [[r0,c0],[r1,c1]] from YAML."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        a, b = value
        return (tuple(a), tuple(b))  # type: ignore[arg-type]
    raise ValueError(f"motion_region must be null or [[r0,c0],[r1,c1]], got: {value}")


def _cfg_from_dict(d: Dict[str, Any]) -> MotionConfig:
    return MotionConfig(
        file_name=d.get("file_name", "recording.tiff"),
        shift_method=d.get("shift_method", "fourier"),
        upsample_factor=int(d.get("upsample_factor", 20)),
        n_parallel_workers=d.get("n_parallel_workers", 4),
        motion_region=_parse_motion_region(d.get("motion_region", None)),
        corrected_basename=d.get("corrected_basename", "motion_corrected.npy"),
    )


def run_motion_stage(
    dataset_id: str,
    session_id: str,
    protocol_list: List[str],
    raw_root: str | Path,
    processed_root: str | Path,
    cfg: Dict[str, Any] | None = None,
):
    cfg = cfg or {}
    mc = _cfg_from_dict(cfg)

    raw_root = Path(raw_root)
    processed_root = Path(processed_root)

    n_total = 0
    for protocol in protocol_list:
        for raw_tiff in _iter_trial_tiffs(raw_root, dataset_id, session_id, protocol, mc.file_name):
            n_total += 1
            trial_dir = raw_tiff.parent
            trial_id = trial_dir.name

            out_dir = processed_root / dataset_id / session_id / protocol / trial_id
            out_dir.mkdir(parents=True, exist_ok=True)

            corrected_path = out_dir / mc.corrected_basename
            motion_vec_path = out_dir / "motion_vectors.npy"
            ref_img_path = out_dir / "motion_reference.png"
            meta_path = out_dir / "motion_metadata.txt"

            print(f"[MOTION] {protocol}/{trial_id}: loading {raw_tiff.name}")
            movie = _load_tiff_stack(raw_tiff).astype(np.float32)

            corrected_movie, motion_vectors, reference_image = apply_motion_correction(
                movie,
                motion_region=mc.motion_region,
                shift_method=mc.shift_method,
                upsample_factor=mc.upsample_factor,
                n_parallel_workers=mc.n_parallel_workers,
                reference_image=None,
            )

            save_corrected_recording(corrected_movie.astype(np.float32), corrected_path)
            save_motion_vectors(motion_vectors, motion_vec_path)
            save_reference_image(reference_image, ref_img_path)
            save_motion_correction_metadata(
                output_path=meta_path,
                shift_method=mc.shift_method,
                n_frames=int(movie.shape[0]),
                image_shape=(int(movie.shape[1]), int(movie.shape[2])),
                raw_trial_path=trial_dir,
                upsample_factor=mc.upsample_factor,
                n_reference_frames=30,
                motion_region=mc.motion_region,
            )

            print(f"[MOTION] saved → {out_dir}")

    if n_total == 0:
        print(f"[MOTION] No TIFF files found for protocols={protocol_list} using file_name='{mc.file_name}'.")
