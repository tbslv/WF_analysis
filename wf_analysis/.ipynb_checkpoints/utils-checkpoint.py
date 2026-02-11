
import tempfile
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np


def discover_animal_sessions(
    base_dir: Union[str, Path],
    session_substring: Optional[str] = None,
    required_file_name: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Scan the base directory and automatically create a mapping:

        {
            "JPCM-08699": ["250905_leica", "250908_leica"],
            "JPCM-08707": ["250905_leica"],
            ...
        }

    Only folders are considered; files are ignored.

    Parameters
    ----------
    base_dir : str or Path
        Root directory containing dataset_id folders.

    session_substring : str, optional
        If given, only sessions whose folder name contains this substring
        (e.g. "leica") will be included.

    required_file_name : str, optional
        If given, the session will only be included if somewhere under
        that session folder there exists a file with this exact name
        (e.g. "recording.tiff"). The search is recursive.

    Returns
    -------
    dict
        Mapping: dataset_id → sorted list of session_ids
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    animal_sessions: Dict[str, List[str]] = {}

    for animal_folder in base_dir.iterdir():
        if not animal_folder.is_dir():
            continue

        dataset_id = animal_folder.name
        sessions: List[str] = []

        for session_folder in animal_folder.iterdir():
            if not session_folder.is_dir():
                continue

            session_name = session_folder.name

            # Filter by substring in session name (e.g. only "leica")
            if session_substring is not None and session_substring not in session_name:
                continue

            # If required_file_name is set, check that it exists somewhere below
            if required_file_name is not None:
                found = any(
                    p.name == required_file_name
                    for p in session_folder.rglob(required_file_name)
                )
                if not found:
                    # skip sessions that don't have the required data
                    continue

            sessions.append(session_name)

        if sessions:
            animal_sessions[dataset_id] = sorted(sessions)

    return animal_sessions





def group_trials_by_stimulus(
    res_list,
    dffs,
    start,
    end,
    stim_reducer=np.mean,
    precision=0, 
):
    """
    Memory-safe grouping of trials by stimulus.
    Computes mean DFF incrementally (no stacking).

    Returns
    -------
    groups : dict
        stim_value -> {
            "indices": np.ndarray,
            "dff_mean": np.ndarray
        }
    """

    stim_values = np.array(
        [stim_reducer(tr["stim"][start:end]) for tr in res_list],
        dtype=float,
    )
    stim_values = np.round(stim_values, precision)

    groups = {}

    for stim in np.unique(stim_values):
        idx = np.where(stim_values == stim)[0]

        mean_dff = None
        n = 0

        for i in idx:
            dff = dffs[i]  # (T, H, W)

            if mean_dff is None:
                mean_dff = np.zeros_like(dff, dtype=np.float32)

            mean_dff += dff
            n += 1

        mean_dff /= n

        groups[stim] = {
            "indices": idx,
            "dff_mean": mean_dff,
        }

    return groups

