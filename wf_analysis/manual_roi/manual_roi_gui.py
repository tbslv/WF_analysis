import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import tkinter as tk
from tkinter import messagebox


def select_single_circle_roi(
    image: np.ndarray,
    radius: int,
    output_dir: Path,
    roi_name: str,
):
    """
    Interactive manual ROI selection.

    Controls
    --------
    Left click : set / move ROI center
    Key 'm'    : save ROI (confirmation popup)
    Key 'r'    : rotate image 90° clockwise
    """

    # -------------------------
    # Backend safety check
    # -------------------------
    backend = matplotlib.get_backend().lower()
    if "inline" in backend:
        raise RuntimeError(
            "Interactive ROI selection requires a GUI backend.\n"
            "Run `%matplotlib qt` or `%matplotlib tk` and restart the kernel."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"ROI_{roi_name}_manual.txt"

    # -------------------------
    # State
    # -------------------------
    current_image = image.copy()
    rotation_k = 0  # number of 90° clockwise rotations
    marker = None
    roi_coords = {"x": None, "y": None}

    # -------------------------
    # Figure setup
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    img_handle = ax.imshow(current_image, cmap="viridis",vmin=0,vmax=.05)
    ax.set_title("Click ROI | 'r' rotate | 'm' save")
    ax.axis("off")

    # -------------------------
    # Mouse click handler
    # -------------------------
    def on_click(event):
        nonlocal marker

        if event.inaxes != ax:
            return

        roi_coords["x"] = int(round(event.xdata))
        roi_coords["y"] = int(round(event.ydata))

        if marker is not None:
            marker.remove()

        marker = ax.scatter(
            roi_coords["x"],
            roi_coords["y"],
            s=100,
            c="red",
            edgecolors="white",
        )

        fig.canvas.draw_idle()

    # -------------------------
    # Rotate image handler
    # -------------------------
    def rotate_image():
        nonlocal current_image, rotation_k, marker

        H, W = current_image.shape
        current_image = np.rot90(current_image, k=-1)  # clockwise
        rotation_k = (rotation_k + 1) % 4

        img_handle.set_data(current_image)

        # Rotate existing ROI marker if present
        if roi_coords["x"] is not None:
            old_x, old_y = roi_coords["x"], roi_coords["y"]
            new_x = H - 1 - old_y
            new_y = old_x
            roi_coords["x"], roi_coords["y"] = new_x, new_y

            if marker is not None:
                marker.remove()
                marker = ax.scatter(
                    new_x,
                    new_y,
                    s=100,
                    c="red",
                    edgecolors="white",
                )

        fig.canvas.draw_idle()

    # -------------------------
    # Key press handler
    # -------------------------
    def on_key(event):
        if event.key == "r":
            rotate_image()
            return

        if event.key != "m":
            return

        if roi_coords["x"] is None:
            print("⚠️ No ROI selected yet.")
            return

        root = tk.Tk()
        root.withdraw()

        answer = messagebox.askyesno(
            title="Save manual ROI",
            message=f"Save ROI at (x={roi_coords['x']}, y={roi_coords['y']})?"
        )

        root.destroy()

        if answer:
            with open(save_path, "w") as f:
                f.write(f"{roi_coords['x']}, {roi_coords['y']}, {radius}\n")

            print(f"✅ Saved manual ROI to:\n{save_path}")
            plt.close(fig)
        else:
            print("❎ ROI not saved. Select a new position.")

    # -------------------------
    # Connect events
    # -------------------------
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
