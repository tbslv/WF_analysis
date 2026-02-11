#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import tempfile
import yaml

from wf_analysis import run_full_pipeline
from wf_analysis.utils import discover_animal_sessions


def run_for_animals_and_sessions(
    base_config_path: Path,
    animal_sessions: dict[str, list[str]],
    run_motion: bool,
    run_mask: bool,
    run_dff: bool,
    run_roi: bool,
    run_traces: bool,
    run_movies: bool,
    start_dataset: str | None = None,
    start_session: str | None = None,
):
    with base_config_path.open("r") as f:
        base_cfg = yaml.safe_load(f)

    started = start_dataset is None and start_session is None

    for dataset_id in sorted(animal_sessions.keys()):
        sessions = sorted(animal_sessions[dataset_id])

        for session_id in sessions:
            if not started:
                if dataset_id < start_dataset:
                    continue
                if dataset_id == start_dataset and start_session is not None:
                    if session_id < start_session:
                        continue
                    started = True
                elif dataset_id > start_dataset:
                    started = True
                if not started:
                    continue

            print("\n" + "=" * 80)
            print(f"Running pipeline for {dataset_id} / {session_id}")
            print("=" * 80)

            cfg = dict(base_cfg)
            cfg["dataset_id"] = dataset_id
            cfg["session_id"] = session_id

            with tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False, mode="w"
            ) as tmp:
                yaml.safe_dump(cfg, tmp)
                tmp_path = Path(tmp.name)

            try:
                run_full_pipeline(
                    tmp_path,
                    run_motion=run_motion,
                    run_mask=run_mask,
                    run_dff=run_dff,
                    run_roi=run_roi,
                    run_traces=run_traces,
                    run_movies=run_movies,
                )
            except Exception as e:
                print(f"[ERROR] Pipeline failed: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Batch WF analysis pipeline (selectable stages)"
    )

    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-b", "--base_dir", required=True)
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--skip-mask", action="store_true")
    parser.add_argument("--skip-dff", action="store_true")
    parser.add_argument("--skip-roi", action="store_true")
    parser.add_argument("--skip-traces", action="store_true")
    parser.add_argument(
        "--run-movies",
        action="store_true",
        help="Generate stimulus-aligned movies",
    )

    parser.add_argument("--start_dataset", default=None)
    parser.add_argument("--start_session", default=None)

    args = parser.parse_args()

    base_config_path = Path(args.config).resolve()
    base_dir = Path(args.base_dir).resolve()

    with base_config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    required_file_name = cfg.get("motion", {}).get("file_name", "recording.tiff")

    animal_sessions = discover_animal_sessions(
        base_dir=base_dir,
        session_substring="leica",
        required_file_name=required_file_name,
    )

    run_for_animals_and_sessions(
        base_config_path=base_config_path,
        animal_sessions=animal_sessions,
        run_motion=not args.skip_motion,
        run_mask=not args.skip_mask,
        run_dff=not args.skip_dff,
        run_roi=not args.skip_roi,
        run_traces=not args.skip_traces,
        run_movies=args.run_movies,
        start_dataset=args.start_dataset,
        start_session=args.start_session,
    )


if __name__ == "__main__":
    main()
