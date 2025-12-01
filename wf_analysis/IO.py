import h5py

def read_trial_data(dataset_id, session_id, base_dir, protocol_list):
    """
    Reads all `data.h5` files across protocols and returns a list of result dicts.

    Each dict has keys like:
    ['camera_sync', 'camera_sync_sr', 'camera_trigger',
     'modality', 'stim', 'stim_sr', 'stim_out', 'stim_out_sr', 'peltier_controller', 'attrs']

    Parameters
    ----------
    dataset_id : str
        e.g. "JPCM-08780"
    session_id : str
        e.g. "251014_leica"
    base_dir : str
        Root directory containing the raw data
    protocol_list : list[str]
        List of protocol folder names

    Returns
    -------
    results : list[dict]
        One dict per trial, containing all relevant data
    """
    results = []
    os.chdir(r"Z:\Individual_Folders\Tobi\WF_axonimaging\axonal_imaging_tobi")

    for protocol_name in protocol_list:
        print(f"\n--- Reading protocol: {protocol_name} ---")
        data_paths = glob.glob(
            f"{base_dir}/{dataset_id}/{session_id}/{protocol_name}/**/data.h5",
            recursive=True
        )
        if not data_paths:
            print(f"⚠️ No data.h5 found for protocol '{protocol_name}'")
            continue

        for h5_path in data_paths:
            trial_id = Path(h5_path).parent.name
            print(f"Reading {trial_id} ({h5_path})")

            res = {}
            with h5py.File(h5_path, "r") as dat_h5:
                # Camera signals
                res["camera_sync"] = np.array(dat_h5["data"]["in"]["camera"])
                res["camera_sync_sr"] = float(dat_h5["data"]["in"]["camera"].attrs["sr"])
                res["camera_trigger"] = np.array(dat_h5["data"]["out"]["camera"])

                # Modality + stimulus
                if "sound" in dat_h5["data"]["out"]:
                    res["modality"] = "sound"
                    res["stim"] = np.array(dat_h5["data"]["out"]["sound"])
                    res["stim_sr"] = float(dat_h5["data"]["out"]["sound"].attrs["sr"])

                elif "touch" in dat_h5["data"]["out"]:
                    res["modality"] = "tactile"
                    res["stim"] = np.array(dat_h5["data"]["out"]["touch"])
                    res["stim_sr"] = float(dat_h5["data"]["out"]["touch"].attrs["sr"])

                elif "temperature" in dat_h5["data"]["in"]:
                    res["modality"] = "temperature"
                    res["stim_out"] = np.array(dat_h5["data"]["out"]["temperature"])
                    res["stim_sr"] = float(dat_h5["data"]["in"]["temperature"].attrs["sr"])
                    res["stim_out_sr"] = float(dat_h5["data"]["out"]["temperature"].attrs["sr"])

                    # Identify peltier controller
                    temp_1sec = res["stim_out"][: int(res["stim_sr"])].mean()
                    if "device" in dat_h5["data"]["in"]["temperature"].attrs:
                        peltier_controller = "qst"
                    else:
                        peltier_controller = "esys" if temp_1sec < 1 else "black"
                    res["peltier_controller"] = peltier_controller

                    # Convert to real temperature
                    if peltier_controller == "esys":
                        res["stim"] = np.array(dat_h5["data"]["in"]["temperature"]) * 4 + 32
                        res["stim_out"] = res["stim_out"] * 4 + 32
                    else:
                        res["stim"] = (
                            np.array(dat_h5["data"]["in"]["temperature"]) * 17.0898 - 5.0176
                        )

                # File attributes
                res["attrs"] = {}
                for key, value in dat_h5.attrs.items():
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    if hasattr(value, "item"):
                        try:
                            value = value.item()
                        except Exception:
                            value = str(value)
                    res["attrs"][key] = value

            results.append(res)

    print(f"✅ Done. Loaded {len(results)} trials total.")
    return results

import os
import glob
from pathlib import Path
import numpy as np

def load_all_dff_files(dataset_id, session_id, protocol_list,
                       processed_root="data/processed",
                       load_arrays=True):
    """
    Loads all dff.npy files created by process_dff_trials.

    Parameters
    ----------
    dataset_id : str
        e.g., "JPCM-08780"
    session_id : str
        e.g., "251014_leica"
    protocol_list : list of str
        List of protocol folder names
    processed_root : str
        Root folder where process_dff_trials saved the data (default: "data/processed")
    load_arrays : bool
        If True, load the actual DFF arrays.
        If False, only return metadata & file paths.

    Returns
    -------
    dff_entries : list of dict
        Each dict contains:
            {
                "protocol": protocol_name,
                "trial_id": folder_name,
                "dff_path": Path object,
                "dff": np.ndarray  (only if load_arrays=True)
            }
    """

    base = Path(processed_root) / dataset_id / session_id
    dff_entries = []

    for protocol_name in protocol_list:
        print(f"\n--- Searching protocol: {protocol_name} ---")
        proto_dir = base / protocol_name

        if not proto_dir.exists():
            print(f"⚠️ Protocol folder missing: {proto_dir}")
            continue

        # Look for: processed/.../protocol_name/*/dff.npy
        dff_paths = sorted(proto_dir.glob("*/dff.npy"))
       

        if not dff_paths:
            print(f"⚠️ No dff.npy files found in {proto_dir}")
            continue

        for dff_path in dff_paths:
            trial_id = dff_path.parent.name

            entry = {
                "protocol": protocol_name,
                "trial_id": trial_id,
                "dff_path": dff_path,
            }

            if load_arrays:
                entry["dff"] = np.load(dff_path)

            dff_entries.append(entry)

    print(f"\nLoaded {len(dff_entries)} DFF files total.")
    return dff_entries


