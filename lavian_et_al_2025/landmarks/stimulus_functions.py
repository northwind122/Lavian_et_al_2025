import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from lavian_et_al_2025.visual_motion.stimulus_functions import stim_vel_dir_dataframe, quantize_directions


HUESHIFT = 2.5


def get_tuning_map_rois(traces, sens_regs, n_dirs=8):
    # calculate positional tuning for extracted traces
    n_t = sens_regs.shape[0]
    reg = sens_regs.T @ traces[:n_t, :]

    # tuning vector
    bin_centers, bins = quantize_directions([0], n_dirs)
    vectors = np.stack([np.cos(bin_centers), np.sin(bin_centers)], 0)
    reg_vectors = vectors @ reg

    angle = np.arctan2(reg_vectors[1], reg_vectors[0])
    amp = np.sqrt(np.sum(reg_vectors ** 2, 0))

    return amp, angle


def get_tuning_map_pixels(img, sens_regs, n_dirs=8):
    # calculate directional tuning from dF/F traces, px-wise
    traces = img.reshape(img.shape[0], -1)

    n_t = sens_regs.shape[0]
    reg = sens_regs.T @ traces[:, :]
    reg = reg.reshape(reg.shape[0], img.shape[-2], img.shape[-1])

    # tuning vector
    bin_centers, bins = quantize_directions([0], n_dirs)
    vectors = np.stack([np.cos(bin_centers), np.sin(bin_centers)], 0)
    reg_vectors = np.reshape(
        vectors @ np.reshape(reg[:, :, :], (n_dirs, -1)),
        (2,) + reg.shape[1:],
    )

    angle = np.arctan2(reg_vectors[1], reg_vectors[0])
    amp = np.sqrt(np.sum(reg_vectors ** 2, 0))

    return amp, angle


def make_sensory_regressors(exp, n_dirs=8, upsampling=5, sampling=1 / 3):
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
