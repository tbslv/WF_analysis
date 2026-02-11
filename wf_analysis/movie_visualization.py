from pathlib import Path
import numpy as np
import imageio


def create_multi_temp_movie(
    dat: dict,
    output_file: Path,
    frames_per_second: float,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
):
    from io import BytesIO
    import matplotlib.pyplot as plt

    movies = dat["movies"]
    traces = dat["traces"]
    trace_sr = dat["trace_sampling_rate"]
    labels = dat.get("labels", None)

    n_cols = len(movies)
    n_frames = movies[0].shape[0]

    fig = plt.figure(figsize=(6 * n_cols, 8))
    gs = fig.add_gridspec(
        2, n_cols,
        height_ratios=[1, 4],
        hspace=0.1,
        wspace=0.05,
    )

    fig.patch.set_facecolor("black")

    axes_trace = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    axes_img = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]

    cursors = []
    images = []

    for i in range(n_cols):
        trace = traces[i]
        t = np.arange(len(trace)) 

        ax = axes_trace[i]
        ax.plot(t, trace, color="white")
        cursor = ax.axvline(0, color="red")
        cursors.append(cursor)

        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("white")
        if labels:
            ax.set_title(labels[i], color="white")

        movie = movies[i]
        vmin_i = np.nanmin(movie) if vmin is None else vmin
        vmax_i = np.nanmax(movie) if vmax is None else vmax

        img = axes_img[i].imshow(
            movie[0], cmap=cmap, vmin=vmin_i, vmax=vmax_i
        )
        axes_img[i].axis("off")
        images.append(img)

    writer = imageio.get_writer(
        str(output_file),
        fps=frames_per_second,
        codec="libx264",
    )

    for f in range(n_frames):
        t = f / frames_per_second * trace_sr
        for i in range(n_cols):
            images[i].set_data(movies[i][f])
            cursors[i].set_xdata([t])

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)

        frame = imageio.v2.imread(buffer)
        buffer.close()

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        writer.append_data(frame)

    writer.close()
    plt.close(fig)
