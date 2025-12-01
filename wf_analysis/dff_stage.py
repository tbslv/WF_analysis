from pathlib import Path
from typing import List

from .dff_helper import process_dff_trials  # your existing helper
# If you want: from .IO import read_trial_data_or_whatever


def run_dff_stage(
    dataset_id: str,
    session_id: str,
    base_raw_dir: str,
    protocol_list: List[str],
    file_name: str = "recording.tiff",
    frame_start: int = 101,
    frame_end: int = 131,
    fps: int = 20,
    do_downscale: bool = True,
):
    """
    Stage 1:
    - For each protocol & trial:
        * load raw/downsized movie
        * compute DFF movie
        * save mean.npy, dff.npy, mean_image.pdf in processed folder
    - Returns list of DFF arrays in processing order.
    """
    dffs = process_dff_trials(
        dataset_id=dataset_id,
        session_id=session_id,
        base_dir=base_raw_dir,
        protocol_list=protocol_list,
        file_name=file_name,
        frame_start=frame_start,
        frame_end=frame_end,
        fps=fps,
        do_downscale=do_downscale,
    )
    return dffs
