from typing import List
import numpy as np

from .traces_helper import (
    load_roi_for_protocols,
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
    stim_groups: dict[float, dict],
    trial_paths: list | None = None,  # NEW
    roi_radius: float = 10.0,
    trace_basename: str = "roi_trace",
    save_format: str = "npy",
    roi_source: str = "prefer_manual",  
):

    """
    Stage 3 (generalized):
    - Load ROI coordinates per stimulus
    - Extract 1D ROI traces for each trial
    - Save traces into trial folders

    Trace files will be named:
        roi_trace_stim_<stim_value>.npy
    """

    # 1) Map trial index -> trial folder
    if trial_paths is None:
        trial_paths = get_trial_paths_from_dff(
            dataset_id=dataset_id,
            session_id=session_id,
            protocol_name=protocol_list[0],
            processed_root=processed_root,
            file_name="dff.npy",
        )


    # 2) Loop over stimulus groups
    for stim_value, group in stim_groups.items():

        stim_name = f"stim_{stim_value}"

        if roi_source == "manual":
            roi_names = [f"{stim_name}_manual"]
        
        elif roi_source == "auto":
            roi_names = [stim_name]
        
        elif roi_source == "prefer_manual":
            roi_names = [f"{stim_name}_manual", stim_name]
        
        else:
            raise ValueError(
                f"Invalid roi_source='{roi_source}'. "
                "Use 'auto', 'manual', or 'prefer_manual'."
            )
        
        roi_dict = {}
        
        for name in roi_names:
            roi_dict = load_roi_for_protocols(
                dataset_id=dataset_id,
                session_id=session_id,
                protocol_list=protocol_list,
                name=name,
                processed_root=processed_root,
            )
            if roi_dict:
                print(f"✅ Using ROI '{name}' for stimulus {stim_value}")
                break
        
        if not roi_dict:
            print(f"⚠️ No ROI found for stimulus {stim_value}, skipping.")
            continue


        if not roi_dict:
            print(f"⚠️ No ROI found for stimulus {stim_value}, skipping.")
            continue

        # For now: assume single protocol / single ROI
        roi = list(roi_dict.values())[0]

        traces = []
        trial_indices = group["indices"]
        
        # 3) Extract traces
        for idx in trial_indices:
            dff_movie = dffs[idx]  # (T, H, W)
            trace = apply_circular_roi_mask(
                dff_movie,
                roi=roi,
                r=roi_radius,
            )
            traces.append(trace)

        # 4) Save traces
        #print(f'*** save {trace_basename}_stim_{stim_value} traces @ {trial_paths} ***')
        
        save_traces_to_trials(
            trace_list=traces,
            trial_paths=trial_paths,
            idx=trial_indices,
            name=f"{trace_basename}_stim_{stim_value}",
            save_as=save_format,
        )

