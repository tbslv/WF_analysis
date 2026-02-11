from pathlib import Path

def load_roi(
    dataset_id,
    session_id,
    protocol_name,
    name,
    processed_root,
):
    from pathlib import Path

    roi_path = (
        Path(processed_root)
        / dataset_id
        / session_id
        / protocol_name
        / f"ROI_{name}.txt"
    )

    with open(roi_path, "r") as f:
        values = [int(v) for v in f.read().strip().split(",")]
    
    if len(values) == 2:
        x, y = values
    
    elif len(values) == 3:
        x, y, extra = values
        print(
            f"ℹ️ ROI '{roi_path.name}': 3 values detected, "
            "using first 2 as (x, y)"
        )
    
    else:
        raise ValueError(
            f"ROI file '{roi_path}' must contain 2 or 3 integers, "
            f"got {len(values)}"
        )
    
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

def get_trial_paths_from_dff(
    dataset_id,
    session_id,
    protocol_name,
    processed_root,
    file_name="dff.npy",
):
    from pathlib import Path

    base = (
        Path(processed_root)
        / dataset_id
        / session_id
        / protocol_name
    )

    return [p.parent for p in base.rglob(file_name)]


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

