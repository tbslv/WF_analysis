from pathlib import Path

def load_roi(dataset_id,
             session_id,
             protocol_name,
             name="cold",
             processed_root="data/processed"):
    """
    Load an ROI coordinate (x, y) from the text file:
        processed_root/dataset_id/session_id/protocol_name/ROI_{name}/ROI_{name}.txt

    Parameters
    ----------
    dataset_id : str
    session_id : str
    protocol_name : str
    name : str
        ROI label, e.g. "cold", "warm", "stimA"
    processed_root : str or Path

    Returns
    -------
    (x, y) : tuple of int
        ROI center coordinate
    """

    roi_path = (
        Path(processed_root)
        / dataset_id
        / session_id
        / protocol_name
        / f"ROI_{name}.txt"
    )

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    with open(roi_path, "r") as f:
        line = f.read().strip()

    try:
        x_str, y_str = line.split(",")
        x, y = int(x_str), int(y_str)
    except Exception as e:
        raise ValueError(
            f"ROI file is malformed. Expected 'x, y', found: '{line}'"
        ) from e

    return (x, y)

def load_roi_for_protocols(dataset_id,
                           session_id,
                           protocol_list,
                           name="cold",
                           processed_root="data/processed"):
    """
    Load ROI coordinates for multiple protocols.

    Returns
    -------
    roi_dict : dict
        protocol_name -> (x, y)
    """
    roi_dict = {}

    for protocol_name in protocol_list:
        try:
            roi = load_roi(
                dataset_id=dataset_id,
                session_id=session_id,
                protocol_name=protocol_name,
                name=name,
                processed_root=processed_root
            )
            roi_dict[protocol_name] = roi

        except FileNotFoundError:
            print(f"⚠️ ROI for protocol '{protocol_name}' not found, skipping.")
        except Exception as e:
            print(f"⚠️ Error reading ROI for '{protocol_name}': {e}")

    return roi_dict


def get_stimulus_sorted_data(res_list, dffs, threshold=32):
    """
    Classifies trials into 'cold' and 'warm' based on mean stimulus value,
    extracts the corresponding DFF arrays, and computes mean DFF for each group.

    Parameters
    ----------
    res_list : list of dict
        Each element must contain a key 'stim' (stimulus trace).
    dffs : list or array of np.ndarray
        List of DFF movies, each shaped (T, H, W).
    threshold : float
        Temperature threshold separating cold vs warm trials (default = 32°C).

    Returns
    -------
    dff_cold_mean : np.ndarray
        Mean DFF for all cold trials, shape (T, H, W).
    dff_warm_mean : np.ndarray
        Mean DFF for all warm trials, shape (T, H, W).
    cold_trials : np.ndarray
        Indices of cold trials.
    warm_trials : np.ndarray
        Indices of warm trials.
    """

    # --------------------------
    # Classify trials
    # --------------------------
    trial_ids = np.array([
        "warm" if np.mean(tr["stim"]) > threshold else "cold"
        for tr in res_list
    ])

    cold_trials = np.where(trial_ids == "cold")[0]
    warm_trials = np.where(trial_ids == "warm")[0]

    # --------------------------
    # Extract DFF arrays
    # --------------------------
    dff_shape = dffs[0].shape  # assume all same shape
    dffs_cold = np.zeros((len(cold_trials),) + dff_shape)
    dffs_warm = np.zeros((len(warm_trials),) + dff_shape)

    for i, idx in enumerate(cold_trials):
        dffs_cold[i] = dffs[idx]

    for i, idx in enumerate(warm_trials):
        dffs_warm[i] = dffs[idx]

 

    return dffs_cold, dffs_warm, cold_trials, warm_trials

import numpy as np

def apply_circular_roi_mask(arr, roi, r):
    """
    Apply a circular mask to a 2D image or 3D movie and return the mean value(s).

    Parameters
    ----------
    arr : np.ndarray
        2D array (H, W) or 3D array (T, H, W)
    roi : tuple (x, y)
        Center of the circle
    r : float or int
        Radius of the ROI in pixels

    Returns
    -------
    masked_mean : float (for 2D arr) or np.ndarray of shape (T,) for 3D arr
        Mean value inside the ROI circle.
    """

    arr = np.asarray(arr)
    roi_x, roi_y = roi

    # Construct coordinate grid
    if arr.ndim == 2:
        H, W = arr.shape
    elif arr.ndim == 3:
        _, H, W = arr.shape
    else:
        raise ValueError("Input arr must be 2D or 3D")

    yy, xx = np.ogrid[:H, :W]

    # Circular mask (True inside circle)
    mask = (xx - roi_x)**2 + (yy - roi_y)**2 <= r**2

    # 2D case -------------------------------------------------------
    if arr.ndim == 2:
        values = arr[mask]
        return values.mean()

    # 3D case (T, H, W) ---------------------------------------------
    elif arr.ndim == 3:
        T = arr.shape[0]
        mean_ts = np.zeros(T)

        for t in range(T):
            values = arr[t][mask]
            mean_ts[t] = values.mean()

        return mean_ts

import glob
from pathlib import Path

def get_trial_paths_from_dff(dataset_id,
                             session_id,
                             protocol_name,
                             base_dir="data/processed",
                             file_name="dff.npy"):
    """
    Find all dff.npy files in a protocol folder and return their trial directories.

    Folder structure:
        base_dir/dataset_id/session_id/protocol_name/<trial_id>/dff.npy

    Parameters
    ----------
    dataset_id : str
    session_id : str
    protocol_name : str
    base_dir : str
        Root processed directory (default: "data/processed")
    file_name : str
        Name of the file to match (default: "dff.npy")

    Returns
    -------
    trial_paths : list of Path
        Each element is the folder containing the dff file.
    """

    search_pattern = (
        f"{base_dir}/{dataset_id}/{session_id}/{protocol_name}/**/*{file_name}"
    )

    dff_paths = glob.glob(search_pattern, recursive=True)

    # extract parent folder of each dff.npy
    trial_paths = [Path(p).parent for p in dff_paths]

    return trial_paths


import numpy as np
from pathlib import Path

def save_traces_to_trials(trace_list, trial_paths, idx, name="roi_trace", save_as="npy"):
    """
    Save a list of time-series traces to their corresponding trial folders.

    Parameters
    ----------
    trace_list : list of np.ndarray
        List of 1D time series traces. Order matches the order of idx.
    trial_paths : list of Path
        Each element is a WindowsPath/Path pointing to the trial directory.
    idx : array-like of int
        Indices selecting which trial_paths correspond to trace_list entries.
    name : str
        Base name of saved trace files, e.g. "roi_trace"
    save_as : str
        "npy" or "txt"

    Saves each trace to:
        <trial_path>/name.npy or name.txt

    Returns
    -------
    saved_files : list of Path
        Full paths to saved trace files.
    """

    saved_files = []

    if len(trace_list) != len(idx):
        raise ValueError("trace_list length and idx length must match.")

    for trace, trial_index in zip(trace_list, idx):
        trial_dir = Path(trial_paths[trial_index])

        if not trial_dir.exists():
            raise FileNotFoundError(f"Trial folder does not exist: {trial_dir}")

        trial_dir.mkdir(parents=True, exist_ok=True)

        # output file name
        if save_as == "npy":
            out_path = trial_dir / f"{name}.npy"
            np.save(out_path, trace)

        elif save_as == "txt":
            out_path = trial_dir / f"{name}.txt"
            np.savetxt(out_path, trace)

        else:
            raise ValueError("save_as must be 'npy' or 'txt'.")

        saved_files.append(out_path)

    return saved_files

