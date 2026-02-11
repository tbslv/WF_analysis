import numpy as np
from pathlib import Path
from tifffile import imread
from skimage.transform import resize

import numpy as np
import logging


logger = logging.getLogger(__name__)

def downscale_movie(mov):
    """
    Downscale a 3D movie array to 512x512 resolution using block averaging.

    Supports input shapes:
    - (T, 1024, 1024) → (T, 512, 512)
    - (T, 2048, 2048) → (T, 512, 512)
    - (T, 512, 512) → returned as-is

    Returns:
        np.ndarray or bool: Downscaled movie, or False if unsupported.
    """
    T, H, W = mov.shape
    if (H, W) == (1024, 1024):
        return mov.reshape(T, 512, 2, 512, 2).mean(4).mean(2).astype(np.uint16)
    elif (H, W) == (2048, 2048):
        return mov.reshape(T, 512, 4, 512, 4).mean(4).mean(2).astype(np.uint16)
    elif (H, W) == (512, 512):
        return mov  # Already correct size
    else:
        logger.info(f"Unsupported image shape {H}x{W} for downscaling.")
        return False
    
def get_trial_movie(tiff_path, do_downscale=False) -> np.ndarray:
    """
    Load a TIFF movie directly from a given file path.

    Args:
        tiff_path (str or Path): Full path to the TIFF file (e.g., recording.tiff)
        do_downscale (bool): Whether to downscale the movie (e.g. to 512x512)

    Returns:
        np.ndarray: Loaded (and optionally downscaled) movie of shape (T, H, W)
    """
    movie_path = Path(tiff_path)
    print(f"Loading movie from: {movie_path}")
    mov = imread(str(movie_path))  # <-- cast Path to str
    return downscale_movie(mov) if do_downscale else mov.astype(np.float32)
