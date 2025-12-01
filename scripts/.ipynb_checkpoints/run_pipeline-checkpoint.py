#!/usr/bin/env python

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from wf_analysis import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the WF analysis pipeline (DFF → ROI → Traces)."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    run_full_pipeline(config_path)


if __name__ == "__main__":
    main()
