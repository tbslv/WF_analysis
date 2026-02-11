#!/usr/bin/env python

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from wf_analysis import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run WF analysis pipeline (selectable stages)"
    )

    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config file",
    )

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

    args = parser.parse_args()

    run_full_pipeline(
        args.config,
        run_motion=not args.skip_motion,
        run_mask=not args.skip_mask,
        run_dff=not args.skip_dff,
        run_roi=not args.skip_roi,
        run_traces=not args.skip_traces,
        run_movies=args.run_movies,
    )


if __name__ == "__main__":
    main()
