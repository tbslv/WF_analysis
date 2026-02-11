from pathlib import Path
import numpy as np

from .manual_roi_gui import select_single_circle_roi
from .manual_roi_io import save_manual_roi
from .manual_roi_utils import load_mean_dff_for_stimulus


def manual_roi_for_dataset(
    dataset_id: str,
    session_id: str,
    protocol: str,
    stimulus_values: list,
    processed_root: str | Path,
    raw_root: str | Path,
    frame_start: int = 100,
    frame_end: int = 140,
    radius: int = 12,
):
    """
    Guided manual ROI selection for multiple stimulus conditions.

    Iterates over stimulus_values and pauses for manual ROI selection
    at each condition.
    """

    processed_root = Path(processed_root)

    roi_base_dir = (
        processed_root
        / dataset_id
        / session_id
        / protocol
    )
    roi_base_dir.mkdir(parents=True, exist_ok=True)

    for stim in stimulus_values:
        print(f"\n=== Manual ROI for stimulus {stim} ===")

        mean_movie, mean_img, _ = load_mean_dff_for_stimulus(
            dataset_id=dataset_id,
            session_id=session_id,
            protocol=protocol,
            stimulus_value=stim,
            processed_root=processed_root,
            raw_root=raw_root,
            frame_start=frame_start,
            frame_end=frame_end,
        )

        roi_name = f"stim_{stim}"

        roi = select_single_circle_roi(
            image=mean_img,
            radius=radius,
            output_dir=roi_base_dir,
            roi_name=roi_name,
        )

        print(f"✔ Finished stimulus {stim}")
