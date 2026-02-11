import numpy as np
from pathlib import Path

from .IO import read_trial_data


def collect_movies_for_stimulus(
    dffs: list[np.ndarray],
    stim_indices: np.ndarray,
    mode: str = "mean",
) -> np.ndarray:
    """
    Collect movies for a stimulus group.

    Parameters
    ----------
    dffs : list of ndarray
        List of DFF movies (T, H, W)
    stim_indices : ndarray
        Indices of trials belonging to this stimulus
    mode : {"mean", "stack"}
        - "mean": average over trials
        - "stack": return stacked array (N, T, H, W)

    Returns
    -------
    movie : ndarray
        If mode == "mean": (T, H, W)
        If mode == "stack": (N, T, H, W)
    """

    movies = [dffs[i] for i in stim_indices]

    if len(movies) == 0:
        raise ValueError("No trials found for this stimulus group.")

    if mode == "mean":
        return np.mean(movies, axis=0)

    elif mode == "stack":
        return np.stack(movies, axis=0)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_stimulus_traces_for_groups(
    dataset_id: str,
    session_id: str,
    protocol: str,
    raw_root: str | Path,
    stim_groups: dict,
) -> dict:
    """
    Compute mean stimulus trace per stimulus value.
    """

    res_list = read_trial_data(
        dataset_id=dataset_id,
        session_id=session_id,
        base_dir=raw_root,
        protocol_list=[protocol],
    )

    if len(res_list) == 0:
        raise RuntimeError("No trials found when computing stimulus traces.")

    stim_arrays = []
    fallback_len = None

    for entry in res_list:
        stim = entry.get("stim", None)
        if stim is not None:
            stim = np.asarray(stim)
            fallback_len = stim.shape[0]
        stim_arrays.append(stim)

    if fallback_len is None:
        raise RuntimeError("No stimulus traces ('stim') found in any trial.")

    for i, stim in enumerate(stim_arrays):
        if stim is None:
            stim_arrays[i] = np.zeros(fallback_len, dtype=float)

    stim_arrays = np.stack(stim_arrays, axis=0)

    stimulus_traces = {}

    for stim_value, group in stim_groups.items():
        indices = np.asarray(group["indices"], dtype=int)

        if indices.size == 0:
            stimulus_traces[stim_value] = np.zeros(fallback_len, dtype=float)
            continue

        valid_idx = indices[(indices >= 0) & (indices < stim_arrays.shape[0])]
        if valid_idx.size == 0:
            stimulus_traces[stim_value] = np.zeros(fallback_len, dtype=float)
        else:
            stimulus_traces[stim_value] = stim_arrays[valid_idx].mean(axis=0)

    return stimulus_traces
