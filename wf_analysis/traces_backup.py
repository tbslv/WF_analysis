from __future__ import annotations

"""
Plot ROI traces for a given dataset.

Supports:
- auto ROI:    roi_trace_stim_<stim>.npy
- manual ROI:  roi_trace_manual_stim_<stim>.npy
"""

from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math

from wf_analysis.IO import read_trial_data
from wf_analysis.utils import group_trials_by_stimulus


# ============================================================
# Constants
# ============================================================

FRAME_RATE = 20.0
STIM_ONSET_FRAME = 100
T_PRE = 2.0
T_POST = 5.0


# ============================================================
# Helpers
# ============================================================

def _stim_window_frames():
    pre = int(T_PRE * FRAME_RATE)
    post = int(T_POST * FRAME_RATE)
    return STIM_ONSET_FRAME - pre, STIM_ONSET_FRAME + post


def _compute_mean_trace(traces: List[np.ndarray]) -> np.ndarray:
    lengths = {t.shape[0] for t in traces}
    if len(lengths) != 1:
        raise ValueError(f"Traces have different lengths: {lengths}")
    return np.stack(traces, axis=0).mean(axis=0)


def _round_up_nice(x: float) -> float:
    if x <= 0:
        return 0.0

    exp = math.floor(math.log10(x))
    frac = x / (10 ** exp)

    if frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 5:
        nice = 5
    else:
        nice = 10

    return nice * (10 ** exp)


def _trace_filename(
    trace_basename: str,
    stim: float,
    roi_mode: str,
) -> str:
    """
    Construct exact trace filename.
    """
    if roi_mode == "auto":
        return f"{trace_basename}_stim_{stim}.npy"
    elif roi_mode == "manual":
        return f"{trace_basename}_manual_stim_{stim}.npy"
    else:
        raise ValueError("roi_mode must be 'auto' or 'manual'")


# ============================================================
# Trace loaders
# ============================================================

def _load_traces_for_condition(
    processed_root: str | Path,
    dataset_id: str,
    session_id: str | None,
    trace_basename: str,
    stim_value: float,
    roi_mode: str = "auto",
) -> List[np.ndarray]:

    processed_root = Path(processed_root)
    base = processed_root / dataset_id
    if session_id is not None:
        base = base / session_id

    fname = _trace_filename(trace_basename, stim_value, roi_mode)
    paths = sorted(base.glob(f"**/{fname}"))

    print(fname)

    traces = []
    for p in paths:
        tr = np.load(p)
        tr = np.squeeze(tr)
        if tr.ndim == 1:
            traces.append(tr)

    if not traces:
        raise RuntimeError(
            f"No traces found for stim={stim_value} "
            f"(roi_mode='{roi_mode}')"
        )

    return traces


def _load_mean_temperature_traces(
    dataset_id: str,
    session_id: str,
    raw_root: str | Path,
    protocol_list: list[str],
    stim_values: list[float],
    stim_start: int = 5000,
    stim_end: int = 7000,
) -> np.ndarray:

    res_list = read_trial_data(
        dataset_id=dataset_id,
        session_id=session_id,
        base_dir=raw_root,
        protocol_list=protocol_list,
    )

    stim_groups = group_trials_by_stimulus(
        res_list=res_list,
        dffs=[0] * len(res_list),
        start=stim_start,
        end=stim_end,
    )

    temps = [res["stim"] for res in res_list]

    temperatures = np.vstack(
        [
            np.array(temps)[stim_groups[stim]["indices"]].mean(axis=0)
            for stim in stim_values
        ]
    )

    return temperatures


# ============================================================
# Plot: session means cold vs warm
# ============================================================

def plot_session_means_cold_warm(
    dataset_id: str,
    processed_root: str | Path = "data/processed",
    raw_root: str | Path = "data/raw",
    protocol_list: list[str] = ("22_42_interleaved",),
    stim_values: list[float] = (22.0, 42.0),
    trace_basename: str = "roi_trace",
    roi_mode: str = "auto",
    save_dir: str | Path | None = None,
    save: bool = False,
):

    processed_root = Path(processed_root)
    dataset_dir = processed_root / dataset_id

    session_ids = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    if not session_ids:
        raise RuntimeError(f"No sessions found for {dataset_id}")

    all_roi_means = []
    all_temp_means = []

    for sess in session_ids:
        sess_traces = []
        for stim in stim_values:
            traces = _load_traces_for_condition(
                processed_root,
                dataset_id,
                sess,
                trace_basename,
                stim,
                roi_mode,
            )
            sess_traces.append(np.mean(traces, axis=0))

        all_roi_means.append(sess_traces)

        temps = _load_mean_temperature_traces(
            dataset_id,
            sess,
            raw_root,
            list(protocol_list),
            list(stim_values),
        )
        all_temp_means.append(temps)

    all_roi_means = np.array(all_roi_means)
    all_temp_means = np.array(all_temp_means)

    f0, f1 = _stim_window_frames()
    all_roi_means = all_roi_means[:, :, f0:f1]
    t_roi = np.linspace(-T_PRE, T_POST, all_roi_means.shape[-1])

    all_temp_means = all_temp_means[
        :, :, int(5000 - 1000 * T_PRE): int(5000 + 1000 * T_POST)
    ]
    t_temp = np.linspace(-T_PRE, T_POST, all_temp_means.shape[-1])

    colors = ["tab:blue", "tab:red"]

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], hspace=0.15)

    for i, stim in enumerate(stim_values):
        ax_t = fig.add_subplot(gs[0, i])
        ax_r = fig.add_subplot(gs[1, i])

        for s in range(len(session_ids)):
            ax_t.plot(t_temp, all_temp_means[s, i], color=colors[i], alpha=0.3)
            ax_r.plot(t_roi, all_roi_means[s, i], color=colors[i], alpha=0.3)

        ax_t.plot(t_temp, all_temp_means[:, i].mean(axis=0), color=colors[i], lw=3)
        ax_r.plot(t_roi, all_roi_means[:, i].mean(axis=0), color=colors[i], lw=3)

        ax_t.axvline(0, color="k", ls="--")
        ax_r.axvline(0, color="k", ls="--")

        ax_t.set_title(f"{stim} °C | {len(session_ids)} sessions")
        ax_r.set_xlabel("Time (s)")
        ax_r.set_ylabel("ΔF/F")
        ax_r.set_ylim(
            np.round(np.min(all_roi_means), 2),
            np.round(np.max(all_roi_means), 2),
        )

    plt.tight_layout()

    if save:
        save_dir = Path(save_dir or ".") / dataset_id
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"session_means_cold_warm_{roi_mode}.png"
        plt.savefig(out, dpi=300)
        print(f"Saved: {out}")

    plt.show()


# ============================================================
# Plot: stimulus-aligned cold vs warm
# ============================================================

def plot_cold_vs_warm_stim_aligned(
    dataset_id: str,
    session_id: str | None = None,
    processed_root: str | Path = "data/processed",
    raw_root: str | Path = "data/raw",
    protocol_list: list[str] = ("22_42_interleaved",),
    stim_values: list[float] = (22.0, 42.0),
    trace_basename: str = "roi_trace",
    roi_mode: str = "auto",
    save_dir: str | Path | None = None,
    save: bool = False,
):

    processed_root = Path(processed_root)
    dataset_dir = processed_root / dataset_id

    session_ids = (
        [session_id]
        if session_id is not None
        else sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    )

    if not session_ids:
        raise RuntimeError(f"No sessions found for {dataset_id}")

    # --------------------------------------------------
    # Load ROI traces
    # --------------------------------------------------
    all_traces = []

    for stim in stim_values:
        stim_traces = []
        for sess in session_ids:
            traces = _load_traces_for_condition(
                processed_root,
                dataset_id,
                sess,
                trace_basename,
                stim,
                roi_mode,
            )
            stim_traces.extend(traces)

        if not stim_traces:
            raise RuntimeError(f"No ROI traces for stim {stim} ({roi_mode})")

        all_traces.append(np.stack(stim_traces, axis=0))

    all_traces = np.stack(all_traces, axis=0)  # (n_stim, n_trials, T)

    # --------------------------------------------------
    # Align ROI traces
    # --------------------------------------------------
    f0, f1 = _stim_window_frames()
    all_traces = all_traces[:, :, f0:f1]
    t_roi = np.linspace(-T_PRE, T_POST, all_traces.shape[-1])

    # --------------------------------------------------
    # Load + pool temperature traces
    # --------------------------------------------------
    all_temperatures = []

    for sess in session_ids:
        temps = _load_mean_temperature_traces(
            dataset_id=dataset_id,
            session_id=sess,
            raw_root=raw_root,
            protocol_list=list(protocol_list),
            stim_values=list(stim_values),
        )
        all_temperatures.append(temps)

    all_temperatures = np.stack(all_temperatures, axis=0)  # (n_sessions, n_stim, T)
    temperatures = all_temperatures.mean(axis=0)

    temperatures = temperatures[
        :, int(5000 - 1000 * T_PRE): int(5000 + 1000 * T_POST)
    ]
    t_temp = np.linspace(-T_PRE, T_POST, temperatures.shape[-1])

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    colors = ["tab:blue", "tab:red"]

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], hspace=0.15)

    for i, stim in enumerate(stim_values):
        ax_temp = fig.add_subplot(gs[0, i])
        ax_tr = fig.add_subplot(gs[1, i])

        # --- temperature ---
        ax_temp.plot(t_temp, temperatures[i], color=colors[i], linewidth=2)
        ax_temp.axvline(0, color="k", linestyle="--")
        ax_temp.set_title(f"{stim} °C ({roi_mode})")
        ax_temp.set_ylabel("Temp")
        ax_temp.set_xlim(-T_PRE, T_POST)

        # --- ROI traces ---
        for tr in all_traces[i]:
            ax_tr.plot(t_roi, tr, color=colors[i], alpha=0.25, linewidth=0.8)

        ax_tr.plot(
            t_roi,
            all_traces[i].mean(axis=0),
            color=colors[i],
            linewidth=3,
        )
        ax_tr.axvline(0, color="k", linestyle="--")
        ax_tr.set_xlabel("Time (s)")
        ax_tr.set_ylabel("ΔF/F")
        ax_tr.set_xlim(-T_PRE, T_POST)
        ax_tr.set_ylim(
            np.round(np.min(all_traces), 2),
            np.round(np.max(all_traces), 2),
        )

    plt.tight_layout()

    if save:
        save_dir = Path(save_dir or ".") / dataset_id
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"traces_cold_warm_{roi_mode}.png"
        plt.savefig(out, dpi=300)
        print(f"Saved: {out}")

    plt.show()


def plot_group_session_means_multi_animals(
    groups: dict[str, list[str]],
    processed_root: str | Path = "data/processed",
    raw_root: str | Path = "data/raw",
    protocol_list: list[str] = ("22_42_interleaved",),
    stim_values: list[float] = (22.0, 42.0),
    trace_basename: str = "roi_trace",
    roi_mode: str = "auto",
    save_dir: str | Path = ".",
    save: bool = True,
):
    """
    Plot pooled session means across multiple animals.
    """

    processed_root = Path(processed_root)
    raw_root = Path(raw_root)

    colors = ["tab:blue", "tab:red"]

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, len(groups), height_ratios=[1, 4], hspace=0.15)

    all_roi_vals = []
    all_temp_vals = []
    cached = {}

    for group_name, animals in groups.items():
        roi_means = []
        temp_means = []

        for dataset_id in animals:
            dataset_dir = processed_root / dataset_id
            if not dataset_dir.exists():
                continue

            session_ids = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())

            for sess in session_ids:
                sess_roi = []
                for stim in stim_values:
                    traces = _load_traces_for_condition(
                        processed_root,
                        dataset_id,
                        sess,
                        trace_basename,
                        stim,
                        roi_mode,
                    )
                    sess_roi.append(np.mean(traces, axis=0))
                roi_means.append(sess_roi)

                temps = _load_mean_temperature_traces(
                    dataset_id,
                    sess,
                    raw_root,
                    list(protocol_list),
                    list(stim_values),
                )
                temp_means.append(temps)

        if roi_means:
            roi_means = np.array(roi_means)
            temp_means = np.array(temp_means)

            f0, f1 = _stim_window_frames()
            roi_means = roi_means[:, :, f0:f1]
            temp_means = temp_means[
                :, :, int(5000 - 1000 * T_PRE): int(5000 + 1000 * T_POST)
            ]

            cached[group_name] = (roi_means, temp_means)
            all_roi_
