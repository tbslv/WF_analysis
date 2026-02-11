from pathlib import Path
from typing import List

from .dff_helper import process_dff_trials


def run_dff_stage(
    dataset_id: str,
    session_id: str,
    base_raw_dir: str | Path,
    processed_root: str | Path,
    protocol_list: List[str],
    file_name: str,
    baseline_start: int,
    baseline_end: int,
    response_start: int,
    response_end: int,
    do_downscale: bool,
):
    """
    Stage 1: ΔF/F computation.

    This function is a thin adapter around process_dff_trials(),
    matching the *actual* signature used in dff_helper.py.
    """

    return process_dff_trials(
        dataset_id,
        session_id,
        base_raw_dir,
        processed_root,
        protocol_list,
        file_name,
        baseline_start,
        baseline_end,
        response_start,
        response_end,
        do_downscale,
    )
