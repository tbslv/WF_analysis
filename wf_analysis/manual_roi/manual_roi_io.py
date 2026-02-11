from pathlib import Path
import json
from typing import List, Dict


def save_manual_roi(
    roi: Dict,
    output_path: Path,
):
    """
    Save manual ROI definition.

    roi dict example:
    {
        "type": "circle",
        "x": 123,
        "y": 456,
        "radius": 10,
        "label": "manual_roi_1"
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(roi, f, indent=2)


def load_manual_roi(path: Path) -> Dict:
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def load_all_manual_rois(
    roi_dir: Path,
) -> List[Dict]:
    roi_dir = Path(roi_dir)
    return [
        load_manual_roi(p)
        for p in sorted(roi_dir.glob("*.json"))
    ]
