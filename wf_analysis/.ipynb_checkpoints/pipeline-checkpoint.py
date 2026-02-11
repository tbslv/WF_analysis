import yaml
from pathlib import Path

from .motion_stage import run_motion_stage
from .dff_stage import run_dff_stage
from .roi_stage import run_roi_stage
from .traces_stage import run_traces_stage
from .movie_stage import run_movie_stage
from .IO import load_all_dff_files, read_trial_data
from .utils import group_trials_by_stimulus
from .mask_stage import run_mask_stage



def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def run_full_pipeline(
    config_path: str | Path,
    run_motion: bool = True,
    run_mask: bool = True,
    run_dff: bool = True,
    run_roi: bool = True,
    run_traces: bool = True,
    run_movies: bool = False,
):
    cfg = load_config(config_path)

    dataset_id = cfg["dataset_id"]
    session_id = cfg["session_id"]
    protocol_list = cfg["protocols"]

    raw_root = cfg["paths"]["raw_root"]
    processed_root = cfg["paths"]["processed_root"]

    start = cfg["stimulus"]["start"]
    end = cfg["stimulus"]["end"]

    motion_cfg = cfg.get("motion", {})

    # --------------------------------------------------
    # Stage 0: Motion correction
    # --------------------------------------------------
    if run_motion:
        print("\n=== Stage 0: Motion correction ===")
        run_motion_stage(
            dataset_id=dataset_id,
            session_id=session_id,
            protocol_list=protocol_list,
            raw_root=raw_root,
            processed_root=processed_root,
            cfg=motion_cfg,
        )
    else:
        print("\n=== Stage 0: Motion correction (SKIPPED) ===")

    # --------------------------------------------------
    # Stage 1: Mask application
    # --------------------------------------------------
    if run_mask:
        print("\n=== Stage 1: Mask application ===")
        run_mask_stage(
            dataset_id=dataset_id,
            session_id=session_id,
            protocol_list=protocol_list,
            raw_root=raw_root,
            processed_root=processed_root,
            mask_cfg=cfg["mask"],
        )
    else:
        print("\n=== Stage 1: Mask application (SKIPPED) ===")

    
    # --------------------------------------------------
    # Stage 2: DFF
    # --------------------------------------------------
    if run_dff:
        print("\n=== Stage 2: DFF computation ===")
        dff_cfg = cfg["dff"]

        dffs = run_dff_stage(
            dataset_id=dataset_id,
            session_id=session_id,
            base_raw_dir=raw_root,
            processed_root=processed_root,
            protocol_list=protocol_list,
            file_name=dff_cfg["file_name"],
            baseline_start=dff_cfg["baseline_start"],
            baseline_end=dff_cfg["baseline_end"],
            response_start=dff_cfg["response_start"],
            response_end=dff_cfg["response_end"],
            do_downscale=dff_cfg["downscale"],
        )
    
    else:
        print("\n=== Stage 2: DFF computation (SKIPPED) ===")
        dffs = None

    # --------------------------------------------------
    # Stage 2: ROI
    # --------------------------------------------------
    if run_roi:

        print("\n=== Stage 3: ROI estimation ===")
        

        if dffs is None:
            print("📂 DFF stage skipped → loading existing DFFs from disk")
        
            dff_entries = load_all_dff_files(
                dataset_id=dataset_id,
                session_id=session_id,
                protocol_list=protocol_list,
                processed_root=processed_root,
                load_arrays=True,
            )
            
            dffs = [e["dff"] for e in dff_entries]
            trial_paths = [e["dff_path"].parent for e in dff_entries]
        
            if not dff_entries:
                raise RuntimeError("No existing DFF files found on disk")
        
            # IMPORTANT: keep original trial order
            
        roi_cfg = cfg["roi"]

        stim_groups, _ = run_roi_stage(
            dataset_id=dataset_id,
            session_id=session_id,
            protocol_list=protocol_list,
            processed_root=processed_root,
            base_raw_dir=raw_root,
            frame_start=roi_cfg["frame_start"],
            frame_end=roi_cfg["frame_end"],
            smooth_sigma=roi_cfg["smooth_sigma"],
            percentile=roi_cfg["percentile"],
            dffs=dffs,
            start=start,
            end=end,
        )
    else:
        print("\n=== Stage 3: ROI estimation (SKIPPED) ===")
        stim_groups = None

    # --------------------------------------------------
    # Stage 3: Traces
    # --------------------------------------------------
    if run_traces:
        print("\n=== Stage 4: Trace extraction ===")


        if dffs is None:
            print("📂 DFF stage skipped → loading existing DFFs from disk")
        
            dff_entries = load_all_dff_files(
                dataset_id=dataset_id,
                session_id=session_id,
                protocol_list=protocol_list,
                processed_root=processed_root,
                load_arrays=True,
            )
            
            dffs = [e["dff"] for e in dff_entries]
            trial_paths = [e["dff_path"].parent for e in dff_entries]
        
            if not dff_entries:
                raise RuntimeError("No existing DFF files found on disk")
        
            # IMPORTANT: keep original trial order
            
        
        if stim_groups is None:
            print("📊 ROI stage skipped → recomputing stimulus groups (no ROI estimation)")
        
            res_list = read_trial_data(
                dataset_id=dataset_id,
                session_id=session_id,
                base_dir=raw_root,
                protocol_list=protocol_list,
            )
        
            stim_groups = group_trials_by_stimulus(
                res_list=res_list,
                dffs=dffs,
                start=start,
                end=end,
            )
        trace_cfg = cfg["trace"]

        run_traces_stage(
                dataset_id=dataset_id,
                session_id=session_id,
                protocol_list=protocol_list,
                processed_root=processed_root,
                dffs=dffs,
                trial_paths=trial_paths,   # NEW
                stim_groups=stim_groups,
                roi_radius=cfg["trace"]["radius_pixels"],
                trace_basename=cfg["trace"]["trace_basename"],
                save_format=cfg["trace"]["save_format"],
                roi_source=cfg["trace"].get("roi_source", "prefer_manual"),
            )


    else:
        print("\n=== Stage 4: Trace extraction (SKIPPED) ===")

    # --------------------------------------------------
# Stage 4: Movies
# --------------------------------------------------
    if run_movies:
        print("\n=== Stage 5: Movie generation ===")
    
        run_movie_stage(
            dataset_id=dataset_id,
            session_id=session_id,
            protocol=protocol_list[0],
            processed_root=processed_root,
            raw_root=raw_root,
            dffs=dffs,
            stim_groups=stim_groups,
            stimulus_window=(start, end),
            movie_cfg=cfg["movie"],
        )


    print("\n✅ Pipeline finished successfully.")
