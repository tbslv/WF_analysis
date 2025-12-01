import numpy as np
from numba import jit, prange


# def get_dff_movie(trial, percentile=15):
#     """
#     Calculate df/f of trial.
#     Use cached percentile calculation if available
#     """

#     path_percentile_folder = os.path.join(
#         config["path_res"],
#         trial["dataset"],
#         "wf_dff",
#         "percentile_{}".format(percentile),
#     )

#     path_percentile_trial = os.path.join(
#         path_percentile_folder, "{}.npy".format(trial["trial_id"])
#     )

#     if os.path.exists(path_percentile_trial):
#         f_base = np.load(path_percentile_trial)
#     else:
#         if not os.path.exists(path_percentile_folder):
#             os.makedirs(path_percentile_folder)

#         f_base = calc_movie_percentile(trial["data"]["img"], percentile)

#         np.save(path_percentile_trial, f_base)

#     # # subtract dark noise count
#     # dark = 395
#     # mov = trial['data']['img'].astype(np.float) - dark
#     # # keep 1 as the lowest level
#     # mov[mov < 1] = 1
#     # f_0 = f_base.astype(np.float) - dark
#     # f_0[f_0 < 1] = 1
#     #
#     # return mov / f_0 - 1
#     return trial["data"]["img"] / f_base - 1


@jit(nopython=True, parallel=True)
def calc_movie_percentile(mov, percentile):
    """
    Calculate the percentile of F for each pixel in a movie.
    Executed in parallel by numba.
    """

    f_base = np.zeros_like(mov[0])

    for r in prange(mov.shape[1]):
        for c in prange(mov.shape[2]):
            trace = mov[:, r, c]
            # numba doesn't support axis keyword in np.percentile so
            # it has to remain in loop
            trace_base = np.percentile(trace, percentile)
            # in case of masked img
            if trace_base == 0:
                trace_base = 1
            f_base[r, c] = trace_base

    return f_base


def calc_dff_movie(mov, percentile=15):
    """
    Calculate df/f for a movie based on a given percentile for the movie
    """

    f_base = calc_movie_percentile(mov, percentile)

    return mov / f_base - 1

def calc_dff_movie_minus500(mov, percentile=15):
    """
    Calculate df/f for a movie based on a given percentile for the movie
    """

    f_base = calc_movie_percentile(mov, percentile)

    return mov / f_base - 1


def calc_dff(trace, percentile=15):
    """
    Calculate the df/f for a trace, e.g. from one pixel
    """

    trace_base = np.percentile(trace, percentile)
    dff = trace / trace_base - 1

    return dff
