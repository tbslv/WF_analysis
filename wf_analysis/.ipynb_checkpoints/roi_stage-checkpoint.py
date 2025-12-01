from typing import List, Tuple
import numpy as np

from .IO import read_trial_data      # your helper with read_trial_data(...) 
from .ROI_helper import (
    compute_and_save_roi_for_protocols,
    seperate_trials,
)


def run_roi_stage(
    dataset_id: str,
    session_id: str,
    protocol_list: List[str],
    processed_root: str,
    temp_threshold: float = 32.0,
    name_cold: str = "cold",
    name_warm: str = "warm",
    frame_start: int = 100,
    frame_end: int = 140,
    smooth_sigma: float = 8.0,
    percentile: float = 99.99,
    dffs: list | np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2:
    - Load trial metadata/stimulus
    - Split trials into 'cold' and 'warm' using temp threshold
    - Compute mean DFF for each group
    - Compute & save ROI for cold and warm (per protocol)
    - Return indices of cold/warm trials for use in trace stage.
    """

    # 1) read data for *all* trials across protocols
    res_list = read_trial_data(
        dataset_id=dataset_id,
        session_id=session_id,
        base_dir="data/raw",      # or from config
        protocol_list=protocol_list,
    )

    # 2) classify & compute mean DFF for cold vs warm trials
    #    note: this expects dffs in the same order as res_list
    dff_cold_mean, dff_warm_mean, cold_trials, warm_trials = seperate_trials(
        res_list=res_list,
        dffs=dffs,
        threshold=temp_threshold,
    )

    # 3) compute ROIs and save for each protocol
    if dff_cold_mean is not None:
        compute_and_save_roi_for_protocols(
            dff_mean=dff_cold_mean,
            processed_root=processed_root,
            dataset_id=dataset_id,
            session_id=session_id,
            protocol_list=protocol_list,
            name=name_cold,
            smooth_sigma=smooth_sigma,
            frame_start=frame_start,
            frame_end=frame_end,
            percentile=percentile,
        )

    if dff_warm_mean is not None:
        compute_and_save_roi_for_protocols(
            dff_mean=dff_warm_mean,
            processed_root=processed_root,
            dataset_id=dataset_id,
            session_id=session_id,
            protocol_list=protocol_list,
            name=name_warm,
            smooth_sigma=smooth_sigma,
            frame_start=frame_start,
            frame_end=frame_end,
            percentile=percentile,
        )

    return dff_cold_mean, dff_warm_mean, cold_trials, warm_trials, res_list
