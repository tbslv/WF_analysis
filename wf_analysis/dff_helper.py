import os

#os.chdir(r"C:\Users\tobiasleva\Work\widefield_pic_gp4_3_tst\notebooks\somatotopy\scripts")
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from wf.get_trial_movie_function import  get_trial_movie, downscale_movie
from wf.dff import calc_dff, calc_dff_movie
import h5py


def process_dff_trials(dataset_id, session_id, base_dir, protocol_list, file_name,
                       frame_start=101, frame_end=131, fps=20, do_downscale=True):
    """
    Processes all TIFF trials into ΔF/F arrays, saves results, and returns the DFF stacks.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., "JPCM-08780")
    session_id : str
        Session identifier (e.g., "251014_leica")
    base_dir : str
        Root directory containing the raw TIFFs (e.g., "data/raw")
    protocol_list : list of str
        List of protocols to process
    frame_start, frame_end : int
        Frame range to extract (default: 5s–7s window at 20 fps)
    fps : int
        Frames per second (used only for reference)
    do_downscale : bool
        Whether to downscale movies when loading

    Returns
    -------
    dffs : list of np.ndarray
        List of 3D ΔF/F arrays (T, H, W) for each trial
    """
    os.chdir(r"Z:\Individual_Folders\Tobi\WF_axonimaging\axonal_imaging_tobi")
    def extract_trial_id(tiff_path):
        return Path(tiff_path).parent.name

    dffs = []

    for protocol_name in protocol_list:
        print(f"\n--- Processing protocol: {protocol_name} ---")

        tiff_paths = glob.glob(
            f"{base_dir}/{dataset_id}/{session_id}/{protocol_name}/**/*{file_name}*",
            recursive=True
        )

        if not tiff_paths:
            print(f"No TIFF files found for protocol {protocol_name}")
            continue

        for tiff in tiff_paths:
            print(f"Processing {tiff}")
            trial_id = extract_trial_id(tiff)

            # Load movie (user’s functions assumed available)
            mov = get_trial_movie(tiff, do_downscale=do_downscale)
            dff = calc_dff_movie(mov)

            # Extract window and rotate
            dff_subset = dff[frame_start:frame_end]
            dff_rot = np.rot90(dff_subset, k=-1, axes=(1, 2))
            mn = np.mean(dff_rot, axis=0)

            # Output folder
            output_base = Path("data/processed") / dataset_id / session_id / protocol_name / trial_id
            output_base.mkdir(parents=True, exist_ok=True)

            # Save outputs
            np.save(output_base / "mean.npy", mn)
            np.save(output_base / "dff.npy", dff)

            # Save mean figure
            fig, ax = plt.subplots()
            im = ax.imshow(mn, cmap='inferno')
            ax.set_title(f"Mean image: {trial_id}")
            ax.axis('on')
            plt.colorbar(im, ax=ax, label='dF/F')
            fig.savefig(output_base / "mean_image.pdf")
            plt.close(fig)

            dffs.append(dff)

    print("All trials processed and saved into their own folders.")
    return dffs



