from pathlib import Path
import numpy as np
from typing import Union

from wf_analysis.IO import read_trial_data
from wf_analysis.utils import group_trials_by_stimulus


def load_mean_dff_for_stimulus(
    dataset_id: str,
    session_id: str,
    protocol: str,
    stimulus_value: float,
    processed_root: Union[str, Path],
    raw_root: Union[str, Path],
    frame_start: int = 100,
    frame_end: int = 140,
    stimulus_start: int = 5000,
    stimulus_end: int = 7000,
):
    """
    Load DFF movies for a given protocol and compute:
      1) mean DFF movie across trials for one stimulus
      2) mean DFF image across trials and time window

    Returns
    -------
    mean_movie : np.ndarray  (T, H, W)
    mean_image : np.ndarray  (H, W)
    stim_indices : np.ndarray
    """

    processed_root = Path(processed_root)
    raw_root = Path(raw_root)

    # --------------------------------------------------
    # Load all DFF movies
    # --------------------------------------------------
    proto_dir = processed_root / dataset_id / session_id / protocol

    from wf_analysis.IO import load_all_dff_files

    entries = load_all_dff_files(
        dataset_id=dataset_id,
        session_id=session_id,
        protocol_list=[protocol],
        processed_root=processed_root,
        load_arrays=True,
    )
    
    dffs = [e["dff"] for e in entries]
    
    if not dffs:
        raise RuntimeError("No dff.npy files found on disk")

    if not dffs:
        raise RuntimeError(f"No dff.npy files found in {proto_dir}")

    # --------------------------------------------------
    # Load trial metadata
    # --------------------------------------------------
    res_list = read_trial_data(
        dataset_id=dataset_id,
        session_id=session_id,
        base_dir=raw_root,
        protocol_list=[protocol],
    )

    if len(res_list) != len(dffs):
        raise RuntimeError(
            f"Mismatch: {len(res_list)} trials in metadata, "
            f"{len(dffs)} DFF movies on disk"
        )

    # --------------------------------------------------
    # Group trials by stimulus
    # --------------------------------------------------
    stim_groups = group_trials_by_stimulus(
        res_list=res_list,
        dffs=dffs,
        start=stimulus_start,
        end=stimulus_end,
    )

    if stimulus_value not in stim_groups:
        raise ValueError(
            f"Stimulus {stimulus_value} not found. "
            f"Available: {list(stim_groups.keys())}"
        )

    stim_indices = stim_groups[stimulus_value]["indices"]

    if stim_indices.size == 0:
        raise RuntimeError(f"No trials found for stimulus {stimulus_value}")

    # --------------------------------------------------
    # Compute means
    # --------------------------------------------------
    movies = [dffs[i] for i in stim_indices]
    mean_movie = np.mean(movies, axis=0)  # (T, H, W)
    mean_image = mean_movie[frame_start:frame_end].mean(axis=0)  # (H, W)

    return mean_movie, mean_image, stim_indices
