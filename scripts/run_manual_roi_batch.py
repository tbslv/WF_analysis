#!/usr/bin/env python

import sys
from pathlib import Path
import argparse
import yaml

# Make repo importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from wf_analysis.manual_roi.manual_roi_gui import select_single_circle_roi
from wf_analysis.manual_roi.manual_roi_utils import load_mean_dff_for_stimulus


def run_manual_roi_batch(cfg: dict):
    """
    Batch manual ROI selection across datasets / sessions / stimuli.
    """

    processed_root = Path(cfg["paths"]["processed_root"])
    raw_root = Path(cfg["paths"]["raw_root"])

    frame_start = cfg["stimulus_window"]["frame_start"]
    frame_end = cfg["stimulus_window"]["frame_end"]
    stimulus_start = cfg["stimulus_window"]["stimulus_start"]
    stimulus_end = cfg["stimulus_window"]["stimulus_end"]
    radius = cfg["roi"]["radius"]

    datasets = cfg["datasets"]

    for dataset_id, sessions in datasets.items():
        print("\n############################")
        print(f"### Dataset: {dataset_id}")
        print("############################")

        for session_id, session_cfg in sessions.items():
            protocol = session_cfg["protocol"]
            stimulus_values = session_cfg["stimuli"]

            print(f"\n=== Session: {session_id} | Protocol: {protocol} ===")

            roi_base_dir = (
                processed_root
                / dataset_id
                / session_id
                / protocol
            )
            roi_base_dir.mkdir(parents=True, exist_ok=True)

            for stim in stimulus_values:
                print(f"\n--- Manual ROI for stimulus {stim} ---")

                mean_movie, mean_img, _ = load_mean_dff_for_stimulus(
                    dataset_id=dataset_id,
                    session_id=session_id,
                    protocol=protocol,
                    stimulus_value=stim,
                    processed_root=processed_root,
                    raw_root=raw_root,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    stimulus_start = stimulus_start,
                    stimulus_end = stimulus_end
                )

                roi_name = f"stim_{stim}"

                select_single_circle_roi(
                    image=mean_img,
                    radius=radius,
                    output_dir=roi_base_dir,
                    roi_name=roi_name,
                )

                print(f"✔ Finished stimulus {stim}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch manual ROI selection (interactive)"
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to manual ROI batch YAML config",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_manual_roi_batch(cfg)


if __name__ == "__main__":
    main()
