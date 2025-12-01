from typing import List
import numpy as np

from .traces_helper import (
    load_roi_for_protocols,
    get_stimulus_sorted_data,
    apply_circular_roi_mask,
    get_trial_paths_from_dff,
    save_traces_to_trials,
)


def run_traces_stage(
    dataset_id: str,
    session_id: str,
    protocol_list: List[str],
    processed_root: str,
    dffs: list | np.ndarray,
    res_list: list,
    temp_threshold: float = 32.0,
    roi_radius: float = 10.0,
    name_cold: str = "cold",
    name_warm: str = "warm",
    trace_basename_cold: str = "roi_cold_trace",
    trace_basename_warm: str = "roi_warm_trace",
    save_format: str = "npy",
):
    """
    Stage 3:
    - Load ROI coordinates
    - Split trials into cold/warm again (but now get per-trial DFF lists)
    - Extract 1D ROI traces from each DFF movie
    - Save each trace into its trial folder
    """

    # 1) Get ROIs for each protocol
    roi_cold = load_roi_for_protocols(
        dataset_id=dataset_id,
        session_id=session_id,
        protocol_list=protocol_list,
        name=name_cold,
        processed_root=processed_root,
    )
    roi_warm = load_roi_for_protocols(
        dataset_id=dataset_id,
        session_id=session_id,
        protocol_list=protocol_list,
        name=name_warm,
        processed_root=processed_root,
    )

    # 2) Stimulus-based sorting of trials, but now we want trial-wise dffs
    dffs_cold, dffs_warm, cold_trials, warm_trials = get_stimulus_sorted_data(
        res_list=res_list,
        dffs=dffs,
        threshold=temp_threshold,
    )

    # 3) Map indices -> trial folders from processed dff.npy
    trial_paths = get_trial_paths_from_dff(
        dataset_id=dataset_id,
        session_id=session_id,
        protocol_name=protocol_list[0],  # if multiple protocols: loop or adapt
        base_dir=processed_root,
        file_name="dff.npy",
    )

    # 4) Extract cold traces
    cold_traces = []
    for i, idx in enumerate(cold_trials):
        dff_movie = dffs[idx]              # (T, H, W)
        # Depending on your design: infer protocol for this trial or assume single
        # Here we assume a single protocol or single ROI for simplicity:
        roi = list(roi_cold.values())[0]
        trace = apply_circular_roi_mask(dff_movie, roi=roi, r=roi_radius)
        cold_traces.append(trace)

    # 5) Extract warm traces
    warm_traces = []
    for i, idx in enumerate(warm_trials):
        dff_movie = dffs[idx]
        roi = list(roi_warm.values())[0]
        trace = apply_circular_roi_mask(dff_movie, roi=roi, r=roi_radius)
        warm_traces.append(trace)

    # 6) Save traces into each trial folder
    save_traces_to_trials(
        trace_list=cold_traces,
        trial_paths=trial_paths,
        idx=cold_trials,
        name=trace_basename_cold,
        save_as=save_format,
    )
    save_traces_to_trials(
        trace_list=warm_traces,
        trial_paths=trial_paths,
        idx=warm_trials,
        name=trace_basename_warm,
        save_as=save_format,
    )
