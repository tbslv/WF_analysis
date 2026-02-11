from __future__ import annotations

"""
Plot ROI traces for widefield datasets.

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


def _trace_filename(trace_basename: str, stim: float, roi_mode: str) -> str:
    if roi_mode == "auto":
        return f"{trace_basename}_stim_{stim}.npy"
    elif roi_mode == "manual":
        return f"{trace_basename}_manual_stim_{stim}.npy"
    else:
        raise ValueError("roi_mode must be 'auto' or 'manual'")


# ============================================================
# Loaders
# ============================================================

def _load_traces_for_condition(
    processed_root: str | Path,
    dataset_id: str,
    session_id: str,
    protocol: str,                    # <<< NEW
    trace_basename: str,
    stim_value: float,
    roi_mode: str,
) -> List[np.ndarray]:

    base = (
        Path(processed_root)
        / dataset_id
        / session_id
        / protocol
    )

    fname = _trace_filename(trace_basename, stim_value, roi_mode)
    paths = sorted(base.glob(f"**/{fname}"))

    traces = []
    for p in paths:
        arr = np.load(p).squeeze()
        if arr.ndim == 1:
            traces.append(arr)

    if not traces:
        raise RuntimeError(
            f"No traces found for stim={stim_value} "
            f"(dataset={dataset_id}, session={session_id}, protocol={protocol})"
        )

    return traces


def _load_mean_stimulus_traces(
    dataset_id: str,
    session_id: str,
    raw_root: str | Path,
    protocol_list: list[str],
    stim_values: list[float],
    stim_start: int = 5000,
    stim_end: int = 7000,
) -> np.ndarray:
    """
    Returns stimulus traces shaped (n_stim, T)
    """

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

    stim_traces = [res["stim"] for res in res_list]

    return np.vstack(
        [
            np.array(stim_traces)[stim_groups[stim]["indices"]].mean(axis=0)
            for stim in stim_values
        ]
    )


# ============================================================
# Plot helpers
# ============================================================

def _apply_protocol_style(ax, protocol: str):
    ax.axvspan(0, 2, color="grey", alpha=0.5, zorder=0)


def _protocol_colors(protocol: str, n: int):
    if "tactile" in protocol.lower():
        return ["black"] * n
    if "sound" in protocol.lower():
        return ["green"] * n
    return ["tab:blue", "tab:red"][:n]


def _save_figure(fig, save_dir, dataset_id, protocol, fname):
    out_dir = Path(save_dir) / dataset_id / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / fname, dpi=300)
    print(f"Saved: {out_dir / fname}")


# ============================================================
# Plot: session means (single animal)
# ============================================================

def plot_session_means_cold_warm(
    dataset_id: str,
    processed_root: str | Path,
    raw_root: str | Path,
    protocol_list: list[str],
    stim_values: list[float],
    trace_basename: str = "roi_trace",
    roi_mode: str = "auto",
    save_dir: str | Path = ".",
    save: bool = True,
):

    protocol = protocol_list[0]
    colors = _protocol_colors(protocol, len(stim_values))

    sessions = sorted((Path(processed_root) / dataset_id).iterdir())
    sessions = [s.name for s in sessions if s.is_dir()]

    roi_means, stim_means = [], []

    for sess in sessions:
        roi_means.append([
            np.mean(
                _load_traces_for_condition(
                    processed_root,
                    dataset_id,
                    sess,
                    protocol,
                    trace_basename,
                    stim,
                    roi_mode,
                ),
                axis=0,
            )
            for stim in stim_values
        ])

        stim_means.append(
            _load_mean_stimulus_traces(
                dataset_id, sess, raw_root, protocol_list, stim_values
            )
        )

    roi_means = np.array(roi_means)
    stim_means = np.array(stim_means)

    roi_ylim = (np.min(roi_means), np.max(roi_means))

    f0, f1 = _stim_window_frames()
    roi_means = roi_means[:, :, f0:f1]
    stim_means = stim_means[:, :, int(5000 - 1000 * T_PRE): int(5000 + 1000 * T_POST)]

    t_roi = np.linspace(-T_PRE, T_POST, roi_means.shape[-1])
    t_stim = np.linspace(-T_PRE, T_POST, stim_means.shape[-1])

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, len(stim_values), height_ratios=[1, 4], hspace=0.15)

    for i, stim in enumerate(stim_values):
        ax_s = fig.add_subplot(gs[0, i])
        ax_r = fig.add_subplot(gs[1, i])

        _apply_protocol_style(ax_s, protocol)
        _apply_protocol_style(ax_r, protocol)

        for s in range(len(sessions)):
            ax_s.plot(t_stim, stim_means[s, i], color=colors[i], alpha=0.5)
            ax_r.plot(t_roi, roi_means[s, i], color=colors[i], alpha=0.5)

        ax_s.plot(t_stim, stim_means[:, i].mean(axis=0), color=colors[i], lw=3)
        ax_r.plot(t_roi, roi_means[:, i].mean(axis=0), color=colors[i], lw=3)

        ax_s.axvline(0, color="k", ls="--")
        ax_r.axvline(0, color="k", ls="--")

        ax_r.set_ylim(roi_ylim)

        ax_s.set_ylabel("Stimulus")
        ax_r.set_ylabel("ΔF/F")
        ax_r.set_xlabel("Time (s)")
        ax_s.set_title(f"{stim}")

    plt.tight_layout()

    if save:
        _save_figure(fig, save_dir, dataset_id, protocol,
                     f"session_means_{roi_mode}.png")

    plt.show()


# ============================================================
# Plot: stimulus-aligned individual trials
# ============================================================

def plot_cold_vs_warm_stim_aligned(
    dataset_id: str,
    processed_root: str | Path,
    raw_root: str | Path,
    protocol_list: list[str],
    stim_values: list[float],
    trace_basename: str = "roi_trace",
    roi_mode: str = "auto",
    save_dir: str | Path = ".",
    save: bool = True,
):

    protocol = protocol_list[0]
    colors = _protocol_colors(protocol, len(stim_values))

    sessions = sorted((Path(processed_root) / dataset_id).iterdir())
    sessions = [s.name for s in sessions if s.is_dir()]

    all_traces = []
    for stim in stim_values:
        traces = []
        for sess in sessions:
            traces.extend(
                _load_traces_for_condition(
                    processed_root,
                    dataset_id,
                    sess,
                    protocol,
                    trace_basename,
                    stim,
                    roi_mode,
                )
            )
        all_traces.append(np.stack(traces))

    all_traces = np.stack(all_traces)
    f0, f1 = _stim_window_frames()
    all_traces = all_traces[:, :, f0:f1]
    t = np.linspace(-T_PRE, T_POST, all_traces.shape[-1])

    stim_means = _load_mean_stimulus_traces(
        dataset_id, sessions[0], raw_root, protocol_list, stim_values
    )
    stim_means = stim_means[:, int(5000 - 1000 * T_PRE): int(5000 + 1000 * T_POST)]
    t_stim = np.linspace(-T_PRE, T_POST, stim_means.shape[-1])

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, len(stim_values), height_ratios=[1, 4], hspace=0.15)

    for i, stim in enumerate(stim_values):
        ax_s = fig.add_subplot(gs[0, i])
        ax_r = fig.add_subplot(gs[1, i])

        _apply_protocol_style(ax_s, protocol)
        _apply_protocol_style(ax_r, protocol)

        ax_s.plot(t_stim, stim_means[i], color=colors[i], lw=3)
        ax_s.axvline(0, color="k", ls="--")

        for tr in all_traces[i]:
            ax_r.plot(t, tr, color=colors[i], alpha=0.3)

        ax_r.plot(t, all_traces[i].mean(axis=0), color=colors[i], lw=3)
        ax_r.axvline(0, color="k", ls="--")

        ax_s.set_ylabel("Stimulus")
        ax_r.set_ylabel("ΔF/F")
        ax_r.set_xlabel("Time (s)")
        ax_s.set_title(f"{stim}")

    plt.tight_layout()

    if save:
        _save_figure(fig, save_dir, dataset_id, protocol,
                     f"stim_aligned_{roi_mode}.png")

    plt.show()


# ============================================================
# Plot: group session means (multi-animal)
# ============================================================

def plot_group_session_means_multi_animals(
    groups: dict[str, list[str]],
    processed_root: str | Path,
    raw_root: str | Path,
    protocol_list: list[str],
    stim_values: list[float],
    trace_basename: str = "roi_trace",
    roi_mode: str = "auto",
    save_dir: str | Path = ".",
    save: bool = True,
):

    protocol = protocol_list[0]
    colors = _protocol_colors(protocol, len(stim_values))

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, len(groups), height_ratios=[1, 4], hspace=0.15)

    all_roi_vals, all_stim_vals = [], []
    cached = {}

    for group_name, animals in groups.items():
        roi_means, stim_means = [], []

        for dataset_id in animals:
            sessions = sorted((Path(processed_root) / dataset_id).iterdir())
            sessions = [s.name for s in sessions if s.is_dir()]

            for sess in sessions:
                roi_means.append([
                    np.nanmean(
                        _load_traces_for_condition(
                            processed_root,
                            dataset_id,
                            sess,
                            protocol,
                            trace_basename,
                            stim,
                            roi_mode,
                        ),
                        axis=0,
                    )
                    for stim in stim_values
                ])

                stim_means.append(
                    _load_mean_stimulus_traces(
                        dataset_id, sess, raw_root, protocol_list, stim_values
                    )
                )

        roi_means = np.array(roi_means)
        stim_means = np.array(stim_means)

        f0, f1 = _stim_window_frames()
        roi_means = roi_means[:, :, f0:f1]
        stim_means = stim_means[:, :, int(5000 - 1000 * T_PRE): int(5000 + 1000 * T_POST)]

        cached[group_name] = (roi_means, stim_means)
        all_roi_vals.append(roi_means)
        all_stim_vals.append(stim_means)

    roi_ylim = (min(np.min(a) for a in all_roi_vals),
                max(np.max(a) for a in all_roi_vals))
    stim_ylim = (min(np.min(a) for a in all_stim_vals),
                 max(np.max(a) for a in all_stim_vals))

    for col, (group, (roi_means, stim_means)) in enumerate(cached.items()):
        t_roi = np.linspace(-T_PRE, T_POST, roi_means.shape[-1])
        t_stim = np.linspace(-T_PRE, T_POST, stim_means.shape[-1])

        ax_s = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])

        _apply_protocol_style(ax_s, protocol)
        _apply_protocol_style(ax_r, protocol)

        for i in range(len(stim_values)):
            for s in range(roi_means.shape[0]):
                ax_s.plot(t_stim, stim_means[s, i], color=colors[i], alpha=0.4)
                ax_r.plot(t_roi, roi_means[s, i], color=colors[i], alpha=0.4)

            ax_s.plot(t_stim, stim_means[:, i].mean(axis=0),
                      color=colors[i], lw=3)
            ax_r.plot(t_roi, roi_means[:, i].mean(axis=0),
                      color=colors[i], lw=3)

        ax_s.set_ylim(stim_ylim)
        ax_r.set_ylim(roi_ylim)
        ax_s.set_ylabel("Stimulus")
        ax_r.set_ylabel("ΔF/F")
        ax_r.set_xlabel("Time (s)")
        ax_s.set_title(group)

    plt.tight_layout()

    if save:
        _save_figure(fig, save_dir, "GROUPS", protocol,
                     f"group_session_means_{roi_mode}.png")

    plt.show()
