import numpy as np
import pandas as pd
from bouter.utilities import calc_vel
from scipy.signal import medfilt


def stim_vel_dir_dataframe(exp):
    try:
        t_vel, vel_x = calc_vel(
            np.diff(exp.stimulus_log.bg_x), exp.stimulus_log.t
        )
        t_vel, vel_y = calc_vel(
            np.diff(exp.stimulus_log.bg_y), exp.stimulus_log.t
        )
    except AttributeError:
        try:
            t_vel, vel_x = calc_vel(
                np.diff(exp.stimulus_log.seamless_image_x), exp.stimulus_log.t
            )
            t_vel, vel_y = calc_vel(
                np.diff(exp.stimulus_log.seamless_image_y), exp.stimulus_log.t
            )
        except AttributeError:
            t_vel, vel_x = calc_vel(
                np.diff(exp.stimulus_log.bg_x), exp.stimulus_log.t
            )
            t_vel, vel_y = calc_vel(
                np.diff(exp.stimulus_log.bg_y), exp.stimulus_log.t
            )

    vel_x, vel_y = (medfilt(v) for v in (vel_x, vel_y))
    vel_dir = np.arctan2(vel_y, vel_x)
    vel_amp = np.sqrt(vel_x ** 2 + vel_y ** 2)
    return pd.DataFrame(dict(t=t_vel, theta=vel_dir, vel=vel_amp))


def quantize_directions(angles, n_dirs=8):
    """ Bin angles into wedge bins

    Parameters
    ----------
    angles: array of angles
    n_dirs: number of wedges to split circles into

    Returns
    -------
    bin indices of the angles

    """
    bin_centers = np.arange(n_dirs + 1) * 2 * np.pi / n_dirs
    bin_hw = np.pi / n_dirs
    bins = np.r_[bin_centers - bin_hw, bin_centers[-1] + bin_hw]
    return bin_centers[:n_dirs], (np.digitize(np.mod(angles, np.pi * 2), bins) - 1) % n_dirs