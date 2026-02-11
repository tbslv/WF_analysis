from typing import Dict, List
import numpy as np

from .IO import read_trial_data
from .ROI_helper import compute_and_save_roi_for_protocols
from .utils import group_trials_by_stimulus


def run_roi_stage(
    dataset_id: str,
    session_id: str,
    protocol_list: List[str],
    processed_root: str,
    base_raw_dir: str,
    frame_start: int = 100,
    frame_end: int = 140,
    smooth_sigma: float = 8.0,
    percentile: float = 99.99,
    dffs: list | np.ndarray = None,
    start: int = 0,
    end: int = 1
) -> Dict[float, dict]:
    """
    Stage 2 (generalized):
    - Load trial metadata
    - Group trials by stimulus amplitude
    - Compute mean DFF per stimulus
    - Compute & save ROI per stimulus (per protocol)

    Returns
    -------
    stim_groups : dict
        stim_value -> {
            "indices": np.ndarray,
            "dffs": np.ndarray,
            "dff_mean": np.ndarray
        }
    """

    # 1) Read trial metadata
    res_list = read_trial_data(
        dataset_id=dataset_id,
        session_id=session_id,
        base_dir=base_raw_dir,
        protocol_list=protocol_list,
    )

    # 2) Group trials by stimulus
    stim_groups = group_trials_by_stimulus(
        res_list=res_list,
        dffs=dffs,
        start=start,
        end =end
    )

    # 3) Compute & save ROIs per stimulus
    for stim_value, group in stim_groups.items():
        print(stim_value)
        compute_and_save_roi_for_protocols(
            dff_mean=group["dff_mean"],
            processed_root=processed_root,
            dataset_id=dataset_id,
            session_id=session_id,
            protocol_list=protocol_list,
            name=f"stim_{stim_value}",
            smooth_sigma=smooth_sigma,
            frame_start=frame_start,
            frame_end=frame_end,
            percentile=percentile,
        )

    return stim_groups, res_list
