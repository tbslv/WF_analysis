from pathlib import Path

from .IO import read_trial_data
from .mask_helper import process_single_trial_mask


def run_mask_stage(
    dataset_id: str,
    session_id: str,
    protocol_list: list[str],
    raw_root: str | Path,
    processed_root: str | Path,
    mask_cfg: dict,
):
    """
    Stage 0.5: Brain masking (per trial)

    - Loads movie with priority:
        masked > motion-corrected > raw tiff
    - Computes brain mask via compute_brain_mask_simple()
    - Saves:
        masked.npy
        mask_contour.png
      into processed_root/<dataset>/<session>/<protocol>/<trial_id>/
    """

    raw_root = Path(raw_root)
    processed_root = Path(processed_root)

    # Required
    thr_val = float(mask_cfg["thr_val"])

    # Optional
    offset = int(mask_cfg.get("offset", 0))
    smooth_radius = int(mask_cfg.get("smooth_radius", 2))
    overwrite = bool(mask_cfg.get("overwrite", True))

    # Consistent with other stages: ensures metadata discovery/logging
    read_trial_data(
        dataset_id=dataset_id,
        session_id=session_id,
        base_dir=raw_root,
        protocol_list=protocol_list,
    )

    for protocol in protocol_list:
        raw_proto_dir = raw_root / dataset_id / session_id / protocol

        if not raw_proto_dir.exists():
            print(f"⚠️ Raw protocol folder not found: {raw_proto_dir}")
            continue

        raw_tiffs = sorted(raw_proto_dir.rglob("recording.tiff"))
        if not raw_tiffs:
            print(f"⚠️ No raw TIFFs found under {raw_proto_dir}")
            continue

        print(f"\n=== MASK: {protocol} (N={len(raw_tiffs)}) ===")

        for raw_tiff in raw_tiffs:
            trial_id = raw_tiff.parent.name

            out_dir = (
                processed_root
                / dataset_id
                / session_id
                / protocol
                / trial_id
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            source = process_single_trial_mask(
                raw_tiff_path=raw_tiff,
                out_dir=out_dir,
                thr_val=thr_val,
                offset=offset,
                smooth_radius=smooth_radius,
                overwrite=overwrite,
            )

            print(f"[MASK] {protocol}/{trial_id} (source={source}) → masked.npy")
