import yaml
from pathlib import Path

from .dff_stage import run_dff_stage
from .roi_stage import run_roi_stage
from .traces_stage import run_traces_stage


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def run_full_pipeline(config_path: str | Path):
    cfg = load_config(config_path)

    dataset_id = cfg["dataset_id"]
    session_id = cfg["session_id"]

    raw_root = cfg["paths"]["raw_root"]
    processed_root = cfg["paths"]["processed_root"]
    protocol_list = cfg["protocols"]

    dff_cfg = cfg["dff"]
    roi_cfg = cfg["roi"]
    trace_cfg = cfg["trace"]

    # 1) DFF stage
    print("\n=== Stage 1: DFF computation ===")
    dffs = run_dff_stage(
        dataset_id=dataset_id,
        session_id=session_id,
        base_raw_dir=raw_root,
        protocol_list=protocol_list,
        file_name=dff_cfg.get("file_name", "recording.tiff"),
        frame_start=dff_cfg.get("frame_start", 101),
        frame_end=dff_cfg.get("frame_end", 131),
        fps=dff_cfg.get("fps", 20),
        do_downscale=dff_cfg.get("downscale", True),
    )

    # 2) ROI stage
    print("\n=== Stage 2: ROI estimation ===")
    (
        dff_cold_mean,
        dff_warm_mean,
        cold_trials,
        warm_trials,
        res_list,
    ) = run_roi_stage(
        dataset_id=dataset_id,
        session_id=session_id,
        protocol_list=protocol_list,
        processed_root=processed_root,
        temp_threshold=roi_cfg.get("temp_threshold", 32.0),
        name_cold=roi_cfg.get("name_cold", "cold"),
        name_warm=roi_cfg.get("name_warm", "warm"),
        frame_start=roi_cfg.get("frame_start", 100),
        frame_end=roi_cfg.get("frame_end", 140),
        smooth_sigma=roi_cfg.get("smooth_sigma", 8.0),
        percentile=roi_cfg.get("percentile", 99.99),
        dffs=dffs,
    )

    # 3) Traces stage
    print("\n=== Stage 3: Trace extraction ===")
    run_traces_stage(
        dataset_id=dataset_id,
        session_id=session_id,
        protocol_list=protocol_list,
        processed_root=processed_root,
        dffs=dffs,
        res_list=res_list,
        temp_threshold=roi_cfg.get("temp_threshold", 32.0),
        roi_radius=trace_cfg.get("radius_pixels", 10.0),
        name_cold=roi_cfg.get("name_cold", "cold"),
        name_warm=roi_cfg.get("name_warm", "warm"),
        trace_basename_cold=trace_cfg.get("trace_basename_cold", "roi_cold_trace"),
        trace_basename_warm=trace_cfg.get("trace_basename_warm", "roi_warm_trace"),
        save_format=trace_cfg.get("save_format", "npy"),
    )

    print("\n✅ Pipeline finished successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WF analysis pipeline")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to YAML config file."
    )
    args = parser.parse_args()

    run_full_pipeline(args.config)
