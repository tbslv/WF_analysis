from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_traces_for_condition(
    processed_root: str | Path,
    dataset_id: str,
    trace_basename: str,
    extension: str = "npy",
) -> List[np.ndarray]:
    """
    Helper: load all traces for a given dataset and condition (cold/warm).

    It recursively searches:
        processed_root / dataset_id / ** / *{trace_basename}*.{extension}

    Parameters
    ----------
    processed_root : str or Path
        Root directory of processed data, e.g. "data/processed".
    dataset_id : str
        Animal ID, e.g. "JPCM-08699".
    trace_basename : str
        Base name used when saving traces, e.g. "roi_cold_trace".
    extension : str
        File extension, "npy" or "txt". Only "npy" is implemented here.

    Returns
    -------
    list of np.ndarray
        List of 1D traces.
    """
    processed_root = Path(processed_root)
    dataset_dir = processed_root / dataset_id

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Processed directory for {dataset_id} not found: {dataset_dir}")

    pattern = f"**/*{trace_basename}*.{extension}"
    trace_paths = list(dataset_dir.glob(pattern))

    traces: List[np.ndarray] = []
    for p in trace_paths:
        if extension == "npy":
            arr = np.load(p)
        elif extension == "txt":
            arr = np.loadtxt(p)
        else:
            raise ValueError(f"Unsupported extension: {extension}")

        # Ensure 1D
        arr = np.squeeze(arr)
        if arr.ndim != 1:
            raise ValueError(f"Trace in {p} is not 1D after squeeze (shape={arr.shape}).")
        traces.append(arr)

    return traces


def _compute_mean_trace(traces: List[np.ndarray]) -> np.ndarray:
    """
    Stack traces and compute mean over axis=0.
    Assumes all traces have same length.
    """
    if not traces:
        raise ValueError("No traces provided to compute mean.")

    lengths = {t.shape[0] for t in traces}
    if len(lengths) != 1:
        raise ValueError(f"Traces have different lengths: {lengths}")

    stacked = np.stack(traces, axis=0)  # [N, T]
    return stacked.mean(axis=0)         # [T]


def plot_traces_for_animal(
    dataset_id: str,
    processed_root: str | Path = "data/processed",
    trace_basename_cold: str = "roi_cold_trace",
    trace_basename_warm: str = "roi_warm_trace",
    extension: str = "npy",
    show: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load and plot traces for one animal (dataset_id) across all sessions.

    For each condition (cold / warm):
      - loads all trace files matching *trace_basename* under processed_root/dataset_id
      - plots each trace as a thin line
      - plots the mean trace as a thick line on top

    Parameters
    ----------
    dataset_id : str
        Animal ID, e.g. "JPCM-08699".
    processed_root : str or Path
        Root directory of processed data, e.g. "data/processed".
    trace_basename_cold : str
        Basename for cold traces, e.g. "roi_cold_trace".
    trace_basename_warm : str
        Basename for warm traces, e.g. "roi_warm_trace".
    extension : str
        File extension, "npy" or "txt".
    show : bool
        If True, calls plt.show() at the end.

    Returns
    -------
    (cold_traces, warm_traces) : tuple of lists of np.ndarray
        The loaded traces for each condition.
    """
    processed_root = Path(processed_root)

    # -------------------------
    # 1) Load all cold traces
    # -------------------------
    cold_traces = _load_traces_for_condition(
        processed_root=processed_root,
        dataset_id=dataset_id,
        trace_basename=trace_basename_cold,
        extension=extension,
    )

    # -------------------------
    # 2) Load all warm traces
    # -------------------------
    warm_traces = _load_traces_for_condition(
        processed_root=processed_root,
        dataset_id=dataset_id,
        trace_basename=trace_basename_warm,
        extension=extension,
    )

    # -------------------------
    # 3) Plot cold
    # -------------------------
    if cold_traces:
        mean_cold = _compute_mean_trace(cold_traces)

        plt.figure(figsize=(8, 5))
        for tr in cold_traces:
            plt.plot(tr, linewidth=0.5, alpha=0.4)
        plt.plot(mean_cold, linewidth=3, label="Mean cold")
        plt.title(f"{dataset_id} – Cold traces (N={len(cold_traces)})")
        plt.xlabel("Frame")
        plt.ylabel("ΔF/F (a.u.)")
        plt.legend()
        plt.tight_layout()
    else:
        print(f"[INFO] No cold traces found for {dataset_id}.")

    # -------------------------
    # 4) Plot warm
    # -------------------------
    if warm_traces:
        mean_warm = _compute_mean_trace(warm_traces)

        plt.figure(figsize=(8, 5))
        for tr in warm_traces:
            plt.plot(tr, linewidth=0.5, alpha=0.4)
        plt.plot(mean_warm, linewidth=3, label="Mean warm")
        plt.title(f"{dataset_id} – Warm traces (N={len(warm_traces)})")
        plt.xlabel("Frame")
        plt.ylabel("ΔF/F (a.u.)")
        plt.legend()
        plt.tight_layout()
    else:
        print(f"[INFO] No warm traces found for {dataset_id}.")

    if show:
        plt.show()

    return cold_traces, warm_traces
