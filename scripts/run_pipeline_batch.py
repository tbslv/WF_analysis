#!/usr/bin/env python

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

#!/usr/bin/env python

import argparse
import tempfile
from pathlib import Path

import yaml

from wf_analysis import run_full_pipeline
from wf_analysis.utils import discover_animal_sessions


def run_for_animals_and_sessions(
    base_config_path: Path,
    animal_sessions: dict[str, list[str]],
    start_dataset: str | None = None,
    start_session: str | None = None,
):
    # Load base config once
    with base_config_path.open("r") as f:
        base_cfg = yaml.safe_load(f)

    started = start_dataset is None and start_session is None

    # Iterate animals in sorted order for reproducibility
    for dataset_id in sorted(animal_sessions.keys()):
        sessions = sorted(animal_sessions[dataset_id])

        print("\n" + "#" * 80)
        print(f"### Dataset: {dataset_id}")
        print("#" * 80)

        for session_id in sessions:
            # Handle restart logic
            if not started:
                # We haven't reached the starting point yet
                if dataset_id < start_dataset:
                    # Skip all datasets that come before the start dataset
                    continue
                if dataset_id == start_dataset and start_session is not None:
                    if session_id < start_session:
                        # Skip sessions before the start session in this animal
                        continue
                    # We have reached (or passed) the desired start point
                    started = True
                elif dataset_id > start_dataset:
                    # We've passed the start dataset
                    started = True

                # If still not started after checks, continue
                if not started:
                    continue

            print("\n" + "=" * 80)
            print(f"Running pipeline for {dataset_id} / {session_id}")
            print("=" * 80)

            # Clone base config and override dataset/session
            cfg = dict(base_cfg)
            cfg["dataset_id"] = dataset_id
            cfg["session_id"] = session_id

            # Write a temporary YAML config for this run
            with tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False, mode="w"
            ) as tmp:
                yaml.safe_dump(cfg, tmp)
                tmp_path = Path(tmp.name)

            # Run the pipeline, but don't crash on errors
            try:
                run_full_pipeline(tmp_path)
            except Exception as e:
                print(
                    f"[ERROR] Pipeline failed for {dataset_id} / {session_id}: {e}"
                )
                print("[INFO] Skipping to next session...")
                continue


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the WF analysis pipeline for multiple animals and sessions.\n"
            "Sessions are auto-discovered under base_dir, filtered to those "
            "containing 'leica' in their name and having the required data file."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to base YAML config file (e.g. wf_analysis/config_example.yaml).",
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        required=True,
        help="Base directory containing dataset_id/session folders (e.g. data/raw).",
    )
    parser.add_argument(
        "--start_dataset",
        default=None,
        help="Optional dataset_id at which to start processing (e.g. JPCM-08780).",
    )
    parser.add_argument(
        "--start_session",
        default=None,
        help=(
            "Optional session_id at which to start processing within start_dataset "
            "(e.g. 251019_leica). If only start_dataset is given, will start at its "
            "first valid session."
        ),
    )
    args = parser.parse_args()

    base_config_path = Path(args.config).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()

    if not base_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_config_path}")

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Load base config to know which raw file name is required (e.g. recording.tiff)
    with base_config_path.open("r") as f:
        base_cfg = yaml.safe_load(f)
    dff_cfg = base_cfg.get("dff", {})
    required_file_name = dff_cfg.get("file_name", "recording.tiff")

    # AUTO-GENERATE animal/session mapping:
    #  - only sessions whose name contains "leica"
    #  - only sessions that have the required raw data file somewhere below
    animal_sessions = discover_animal_sessions(
        base_dir=base_dir,
        session_substring="leica",
        required_file_name=required_file_name,
    )

    print("\nDiscovered animals & sessions (filtered):")
    for a, s in sorted(animal_sessions.items()):
        print(f"  {a}: {s}")

    run_for_animals_and_sessions(
        base_config_path=base_config_path,
        animal_sessions=animal_sessions,
        start_dataset=args.start_dataset,
        start_session=args.start_session,
    )


if __name__ == "__main__":
    main()
