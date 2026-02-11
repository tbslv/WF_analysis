import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def compute_and_save_roi_for_protocols(
        dff_mean,
        processed_root,
        dataset_id,
        session_id,
        protocol_list,
        name="stim",
        smooth_sigma=1,
        frame_start=100,
        frame_end=140,
        percentile=99.99):
    """
    Compute 2D ROI (center of hottest pixels) from DFF mean movies for
    multiple protocols, save ROI text + diagnostic plots per protocol.

    Parameters
    ----------
    dff_mean_dict : dict
        Mapping: protocol_name -> dff_mean array (T, H, W).
        Example: {"22_42_interleaved": dff_cold_mean, ...}
    processed_root : str or Path
        Root folder where processed data are stored (e.g. "data/processed").
    dataset_id : str
    session_id : str
    protocol_list : list of str
        List of protocol names to process.
    name : str
        ROI label (e.g. "cold", "warm", "stimA").
    smooth_sigma : float
        Gaussian smoothing sigma.
    frame_start, frame_end : int
        Frames used to compute average activation image.
    percentile : float
        Percentile threshold to detect hot pixels.

    Returns
    -------
    roi_dict : dict
        Mapping protocol_name -> (x, y) ROI center.
    """

    roi_dict = {}

    for protocol_name in protocol_list:
        

        

        # --- compute smoothed activation map ---
        dff_slice = dff_mean[frame_start:frame_end].mean(axis=0)
        smooth_map = gaussian_filter(dff_slice, sigma=smooth_sigma)

        # --- threshold for hot pixels ---
        thr = np.percentile(smooth_map.ravel(), percentile)
        points = np.where(smooth_map > thr)

        if points[0].size == 0:
            print(
                f"⚠️ ROI '{name}' ({protocol_name}): "
                f"No pixels above percentile {percentile}. "
                "Falling back to global maximum."
            )
        
            # fallback: brightest pixel
            flat_idx = np.nanargmax(smooth_map)
            roi_y, roi_x = np.unravel_index(flat_idx, smooth_map.shape)
        
        else:
            roi_x = int(np.mean(points[1]))
            roi_y = int(np.mean(points[0]))
        roi = (roi_x, roi_y)
        roi_dict[protocol_name] = roi

        # --- save location per protocol ---
        processed_root = Path(processed_root)

        base = (
            processed_root
            / dataset_id
            / session_id
            / protocol_name
        )
        base.mkdir(parents=True, exist_ok=True)

        # --- save ROI text file ---
        roi_path = base / f"ROI_{name}.txt"
        with open(roi_path, "w") as f:
            f.write(f"{roi_x}, {roi_y}\n")

        # --- save diagnostic plot ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))

        axes[0].imshow(smooth_map)
        axes[0].set_title(f"{protocol_name} – Smoothed map ({name})")
        axes[0].scatter(points[1], points[0], s=10, color="red")
        axes[0].scatter(roi_x, roi_y, s=60, color="green")

        axes[1].imshow(dff_slice)
        axes[1].set_title(f"{protocol_name} – Raw mean map ({name})")
        axes[1].scatter(points[1], points[0], s=10, color="red")
        axes[1].scatter(roi_x, roi_y, s=60, color="green")

        plt.tight_layout()
        fig.savefig(base / f"ROI_{name}_Estimation.png", dpi=200)
        plt.close(fig)

        print(f"✅ Saved ROI for protocol '{protocol_name}' at {roi_path}")

    return roi_dict


import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_smooth_2d(arr, sigma):
    """
    Apply 2D Gaussian smoothing to an array.

    Handles:
        - 2D arrays  (H, W)
        - 3D arrays  (T, H, W), smoothing only in spatial dimensions

    Parameters
    ----------
    arr : np.ndarray
        Input array, either (H, W) or (T, H, W).
    sigma : float or tuple
        Standard deviation of Gaussian kernel in pixels.
        If float → same sigma for H and W

    Returns
    -------
    smoothed : np.ndarray
        Same shape as input.
    """

    arr = np.asarray(arr)

    if arr.ndim == 2:
        # simple 2D smoothing
        return gaussian_filter(arr, sigma=sigma)

    elif arr.ndim == 3:
        # smooth each frame independently
        T = arr.shape[0]
        out = np.zeros_like(arr)

        for t in range(T):
            out[t] = gaussian_filter(arr[t], sigma=sigma)

        return out

    else:
        raise ValueError("Input must be 2D (H,W) or 3D (T,H,W).")


import numpy as np




