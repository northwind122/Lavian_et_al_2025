import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from lavian_et_al_2025.visual_motion.colors import JCh_to_RGB255
from lavian_et_al_2025.visual_motion.stimulus_functions import quantize_directions


HUESHIFT = 2.5


def get_tuning_map_rois(traces, sens_regs, n_dirs=8):
    """
    Calculates positional tuning properties for traces extracted from ROIs.

    For each ROI, computes the preferred landmark position and tuning amplitude
    by projecting neural activity onto positional regressors.

    Parameters
    ----------
    traces : ndarray
        Neural activity traces with shape (time, ROIs)
    sens_regs : ndarray or DataFrame
        Sensory regressor matrix containing landmark position information
        with shape (time, n_positions)
    n_dirs : int, default=8
        Number of position bins (evenly spaced around the arena)

    Returns
    -------
    amp : ndarray
        Tuning strength for each ROI - indicates how strongly the neuron
        responds to landmark position
    angle : ndarray
        Preferred landmark position (in radians) for each ROI
    """

    n_t = sens_regs.shape[0]
    reg = sens_regs.T @ traces[:n_t, :]

    # tuning vector
    bin_centers, bins = quantize_directions([0], n_dirs)
    vectors = np.stack([np.cos(bin_centers), np.sin(bin_centers)], 0)
    reg_vectors = vectors @ reg

    # Extract angle (preferred position) and amplitude (tuning strength)
    angle = np.arctan2(reg_vectors[1], reg_vectors[0])
    amp = np.sqrt(np.sum(reg_vectors ** 2, 0))

    return amp, angle


def get_tuning_map_pixels(img, sens_regs, n_dirs=8):
    """
    Calculates positional tuning properties for each pixel in an imaging dataset.

    This function creates a spatial map by computing the preferred landmark position and tuning amplitude for each pixel
    in an imaging stack by projecting pixel activity onto positional regressors.

    Parameters
    ----------
    img : ndarray
        Imaging data with shape (time, x, y) representing activity over time for each pixel
    sens_regs : ndarray or DataFrame
        Sensory regressor matrix containing landmark position information with shape (time, n_positions)
    n_dirs : int, default=8
        Number of position bins (evenly spaced around the arena)

    Returns
    -------
    amp : ndarray
        array of tuning amplitude for each pixel
    angle : ndarray
        array of preferred landmark position (in radians) for each pixel
    """
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

    # Extract angle (preferred position) and amplitude (tuning strength)
    angle = np.arctan2(reg_vectors[1], reg_vectors[0])
    amp = np.sqrt(np.sum(reg_vectors ** 2, 0))

    return amp, angle


def color_stack(
        amp,
        angle,
        hueshift=2.5,
        amp_percentile=80,
        maxsat=50,
        lightness_min=100,
        lightness_delta=-40,
    ):
    """
    This function converts amplitude and angle data to RGB color values. This function takes
    amplitude and angle arrays and maps them to colors where:
    - Amplitude controls brightness and saturation
    - Angle controls the hue (color)

    Parameters
    ----------
    amp : numpy.ndarray
        Array of amplitude values. Higher values will appear more saturated and
        typically darker (depending on lightness_delta).
    angle : numpy.ndarray
        Array of angular values in radians. These determine the hue of the output colors.
    hueshift : float, optional
        Constant offset added to angles before converting to hue (in radians).
    amp_percentile : float, optional
        Percentile of amplitude values to use for normalization (0-100).
    maxsat : float, optional
        Maximum saturation value in LCh color space.
    lightness_min : float, optional
        Base lightness value in LCh color space.
    lightness_delta : float, optional
        Change in lightness as amplitude increases from 0 to maximum.
        Negative values make higher amplitudes darker. Default is -40.

    Returns
    -------
    numpy.ndarray
        Array of RGB values with shape (n, 3) where n is the length of the input arrays.
        Values are scaled to 0-255 range (8-bit RGB).
    """

    output_lch = np.zeros((amp.shape[0], 3))
    maxamp = np.percentile(amp, amp_percentile)

    output_lch[:, 0] = (
            lightness_min + (np.clip(amp / maxamp, 0, 1)) * lightness_delta
    )
    output_lch[:, 1] = (np.clip(amp / maxamp, 0, 1)) * maxsat
    output_lch[:, 2] = (angle + hueshift) * 180 / np.pi

    return JCh_to_RGB255(output_lch)

