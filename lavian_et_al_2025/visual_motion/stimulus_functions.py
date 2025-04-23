import numpy as np
import pandas as pd
from bouter.utilities import calc_vel
from scipy.interpolate import interp1d
from scipy.signal import convolve2d, medfilt
import colorspacious


N_DIRS = 8
HUESHIFT = 2.5


def make_sensory_regressors(exp, n_dirs=8, upsampling=5, sampling= 1/3):

    stim = stim_vel_dir_dataframe(exp)
    bin_centres, dir_bins = quantize_directions(stim.theta)
    ind_regs = np.zeros((n_dirs, len(stim)))
    for i_dir in range(n_dirs):
        ind_regs[i_dir, :] = (np.abs(dir_bins - i_dir) < 0.1) & (stim.vel > 0.1)

    dt_upsampled = sampling / upsampling
    t_imaging_up = np.arange(0, stim.t.values[-1], dt_upsampled)
    reg_up = interp1d(stim.t.values, ind_regs, axis=1, fill_value="extrapolate")(
        t_imaging_up
    )

    # 6s kernel
    u_steps = t_imaging_up.shape[0]
    u_time = np.arange(u_steps) * dt_upsampled
    decay = np.exp(-u_time / (1.5 / np.log(2)))
    kernel = decay / np.sum(decay)

    convolved = convolve2d(reg_up, kernel[None, :])[:, 0:u_steps]
    reg_sensory = convolved[:, ::upsampling]

    return pd.DataFrame(reg_sensory.T, columns=[f"motion_{i}" for i in range(n_dirs)])


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
    bin_centers = np.arange(n_dirs+1)*2*np.pi/n_dirs
    bin_hw = np.pi/n_dirs
    bins = np.r_[bin_centers-bin_hw, bin_centers[-1]+bin_hw]
    return bin_centers[:n_dirs],  (np.digitize(np.mod(angles, np.pi*2), bins) - 1) % n_dirs


def get_paint_function(stimulus_log, protocol_name):
    '''
    :param stimulus_log: the stimulus log saved by stytra
    :param protocol_name: the name of the protocol taken from the stytra metadata file
    :return:
    s = 3 x n matrix of RGB values representing the values of the stimulus
    t = n x 2 matrix of time points for stimulus start (1st column) and end (2nd column)
    '''
    print(stimulus_log.columns)
    paint_function = stimulus_coloring_mapping.get(protocol_name)
    s, t = paint_function(stimulus_log)
    return s, t


def stimulus_plot_e0040(stimulus_log):
    t_change, t_change_ind = get_timing_from_current_phase(stimulus_log)[0:2]

    sx = np.diff(np.asarray(stimulus_log.bg_x))[t_change_ind[:, 0]]
    sy = np.diff(np.asarray(stimulus_log.bg_y))[t_change_ind[:, 0]]
    ind0 = np.where((sx == 0) & (sy == 0))[0]

    angles = np.arctan2(sy, sx)
    angles = np.stack(
        [
            np.full(len(sx), 60),
            np.full(len(sx), 60),
            (-angles + 2.5) * 180 / np.pi,
        ],
        1)
    s = JCh_to_RGB255(angles)
    s[ind0, :] = 0

    return s, t_change


def JCh_to_RGB255(x):
    output = np.clip(colorspacious.cspace_convert(x, "JCh", "sRGB1"), 0, 1)
    return (output * 255).astype(np.uint8)


def get_tuning_map(traces, sens_regs, n_dirs=8):
    # calculate directional tuning from zscored traces for each roi
    n_t = sens_regs.shape[0]
    reg = sens_regs.values.T @ traces[:n_t, :]

    # tuning vector
    bin_centers, bins = quantize_directions([0], n_dirs)
    vectors = np.stack([np.cos(bin_centers), np.sin(bin_centers)], 0)
    reg_vectors = vectors @ reg

    angle = np.arctan2(reg_vectors[1], reg_vectors[0])
    amp = np.sqrt(np.sum(reg_vectors ** 2, 0))

    return amp, angle

