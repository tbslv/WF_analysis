from pathlib import Path
import numpy as np

from .movie_helper import (
    collect_movies_for_stimulus,
    compute_stimulus_traces_for_groups,
)
from .movie_visualization import create_multi_temp_movie
from .utils import group_trials_by_stimulus
from .IO import read_trial_data


def _load_all_dffs(
    dataset_id: str,
    session_id: str,
    protocol: str,
    processed_root: str | Path,
):
    processed_root = Path(processed_root)
    proto_dir = processed_root / dataset_id / session_id / protocol

    dffs = []
    for dff_path in sorted(proto_dir.glob("*/dff.npy")):
        dffs.append(np.load(dff_path))

    if not dffs:
        raise RuntimeError(f"No dff.npy files found in {proto_dir}")

    return dffs


def _load_stim_groups_from_disk(
    dataset_id: str,
    session_id: str,
    protocol: str,
    raw_root: str | Path,
    dffs: list[np.ndarray],
    start: int,
    end: int,
):
    res_list = read_trial_data(
        dataset_id=dataset_id,
        session_id=session_id,
        base_dir=raw_root,
        protocol_list=[protocol],
    )

    return group_trials_by_stimulus(
        res_list=res_list,
        dffs=dffs,
        start=start,
        end=end,
    )


def run_movie_stage(
    dataset_id: str,
    session_id: str,
    protocol: str,
    processed_root: str | Path,
    raw_root: str | Path,
    dffs: list[np.ndarray] | None,
    stim_groups: dict | None,
    stimulus_window: tuple[int, int],
    movie_cfg: dict,
):
    """
    Movie stage – fully configurable via YAML.
    """

    # -------------------------
    # Read config
    # -------------------------
    output_name = movie_cfg.get("output_name", "stimulus_comparison.mp4")
    cmap = movie_cfg.get("cmap", "inferno")
    vmin = movie_cfg.get("vmin", None)
    vmax = movie_cfg.get("vmax", None)
    fps = movie_cfg.get("frames_per_second", 20)
    trace_sr = movie_cfg.get("sampling_rate", 1000)

    # -------------------------
    # Load missing inputs
    # -------------------------
    if dffs is None:
        print("[MOVIE] Loading DFFs from disk")
        dffs = _load_all_dffs(
            dataset_id, session_id, protocol, processed_root
        )

    if stim_groups is None:
        print("[MOVIE] Recomputing stim_groups from disk")
        stim_groups = _load_stim_groups_from_disk(
            dataset_id=dataset_id,
            session_id=session_id,
            protocol=protocol,
            raw_root=raw_root,
            dffs=dffs,
            start=stimulus_window[0],
            end=stimulus_window[1],
        )

    # -------------------------
    # Compute stimulus traces
    # -------------------------
    stimulus_traces = compute_stimulus_traces_for_groups(
        dataset_id=dataset_id,
        session_id=session_id,
        protocol=protocol,
        raw_root=raw_root,
        stim_groups=stim_groups,
    )

    # -------------------------
    # Assemble movies
    # -------------------------
    movies = []
    traces = []
    labels = []

    for stim_value, group in stim_groups.items():
        movie = collect_movies_for_stimulus(
            dffs=dffs,
            stim_indices=group["indices"],
            mode="mean",
        )

        movies.append(movie)
        traces.append(stimulus_traces[stim_value])
        labels.append(f"stim={stim_value}")

    dat = {
        "movies": movies,
        "traces": traces,
        "trace_sampling_rate": trace_sr,
        "labels": labels,
    }

    out_dir = (
        Path(processed_root)
        / dataset_id
        / session_id
        / protocol
        / "movies"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / output_name

    create_multi_temp_movie(
        dat=dat,
        output_file=output_file,
        frames_per_second=fps,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    print(f"🎥 Saved movie: {output_file}")
