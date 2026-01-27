import numpy as np
import pandas as pd
from bouter.utilities import calc_vel
from scipy.interpolate import interp1d
from scipy.signal import convolve2d, medfilt
from lavian_et_al_2025.visual_motion.colors import JCh_to_RGB255


N_DIRS = 8
HUESHIFT = 2.5


def make_sensory_regressors(exp, n_dirs=8, upsampling=5, sampling= 1/3):
    """
        Creates directional motion regressors from experiment data.

        This function quantizes motion direction into discrete bins, creates variables for each direction
        and applies an exponential decay kernel,

        Parameters
        ----------
        exp : object
            Experiment object containing stimulus log data
        n_dirs : int, default=8
            Number of direction bins to use (evenly spaced around 360 degrees)
        upsampling : int, default=5
            Factor by which to upsample the data for smoother convolution
        sampling : float, default=1/3
            imaging sampling rate in Hz

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each direction regressor
        """
    # Extract stimulus velocity and direction data
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

    # Create exponential decay kernel (half-life of 1.5s)
    u_steps = t_imaging_up.shape[0]
    u_time = np.arange(u_steps) * dt_upsampled
    decay = np.exp(-u_time / (1.5 / np.log(2)))
    kernel = decay / np.sum(decay)

    # Convolve with kernel and downsample
    convolved = convolve2d(reg_up, kernel[None, :])[:, 0:u_steps]
    reg_sensory = convolved[:, ::upsampling]

    return pd.DataFrame(reg_sensory.T, columns=[f"motion_{i}" for i in range(n_dirs)])


def stim_vel_dir_dataframe(exp):
    """
    Extracts stimulus velocity and direction information from experiment stimulus_log data.

    This function calculates velocity components from position data in the stimulus log
    and computes direction and velocity.

    Parameters
    ----------
    exp : object
        Experiment object containing stimulus log data

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for time (t), direction (theta), and velocity (vel)
    """

    # Try different attribute names that might exist for stimulus position
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

    # Calculate motion direction (in radians) from x and y components
    vel_x, vel_y = (medfilt(v) for v in (vel_x, vel_y))
    vel_dir = np.arctan2(vel_y, vel_x)

    # Calculate motion speed
    vel_amp = np.sqrt(vel_x ** 2 + vel_y ** 2)
    return pd.DataFrame(dict(t=t_vel, theta=vel_dir, vel=vel_amp))


def quantize_directions(angles, n_dirs=8):
    """ Bins continuous angle values into discrete direction categories.

    Parameters
    ----------
    angles : array-like
        Array of angles in radians
    n_dirs : int, default=8
        Number of direction bins to divide the circle into

    Returns
    -------
    bin_centers : ndarray
        Angular positions (in radians) of the bin centers
    bin_indices : ndarray
        Bin index (0 to n_dirs-1) for each input angle

    """
    bin_centers = np.arange(n_dirs+1)*2*np.pi/n_dirs
    bin_hw = np.pi/n_dirs
    bins = np.r_[bin_centers-bin_hw, bin_centers[-1]+bin_hw]
    return bin_centers[:n_dirs],  (np.digitize(np.mod(angles, np.pi*2), bins) - 1) % n_dirs


def get_tuning_map_rois(traces, sens_regs, n_dirs=8):
    """
        Calculates directional tuning properties for neural responses.

        For each ROI, computes the amplitude and preferred angle of directional
        tuning by projecting neural activity onto directional regressors.

        Parameters
        ----------
        traces : ndarray
            Neural activity traces, shape (time, ROIs)
        sens_regs : DataFrame
            Sensory regressor matrix from make_sensory_regressors
        n_dirs : int, default=8
            Number of direction bins

        Returns
        -------
        amp : ndarray
            Tuning strength for each ROI
        angle : ndarray
            Preferred direction (in radians) for each ROI
        """

    n_t = sens_regs.shape[0]

    # Project neural activity onto direction regressors
    reg = sens_regs.values.T @ traces[:n_t, :]

    # Get direction bin centers and convert to vectors
    bin_centers, bins = quantize_directions([0], n_dirs)
    vectors = np.stack([np.cos(bin_centers), np.sin(bin_centers)], 0)
    reg_vectors = vectors @ reg

    # Extract angle (preferred direction) and amplitude (tuning strength)
    angle = np.arctan2(reg_vectors[1], reg_vectors[0])
    amp = np.sqrt(np.sum(reg_vectors ** 2, 0))

    return amp, angle


def get_tuning_map_pixels(img, sens_regs, n_dirs=8):
    """
    Calculates directional tuning properties for each pixel in an imaging dataset.

    This function computes the preferred direction and tuning amplitude for each pixel
    in an imaging stack by projecting pixel activity onto directional motion regressors.
    It performs a pixel-wise version of the ROI-based tuning analysis.

    Parameters
    ----------
    img : ndarray
        Imaging data with shape (time, x, y) representing fluorescence
        over time for each pixel
    sens_regs : DataFrame
        Sensory regressor matrix from make_sensory_regressors with shape (time, n_directions)
    n_dirs : int, default=8
        Number of direction bins used in the regressors

    Returns
    -------
    amp : ndarray
        2D array (height, width) of tuning amplitude for each pixel
    angle : ndarray
        2D array (height, width) of preferred direction (in radians) for each pixel
        """

    # reshape imaging data to (time, pixels) for matrix operations
    traces = img.reshape(img.shape[0], -1)

    n_t = sens_regs.shape[0]
    reg = sens_regs.values.T @ traces[:n_t, :]
    reg = reg.reshape(reg.shape[0], img.shape[-2], img.shape[-1])

    # Get direction bin centers and convert to unit vectors
    bin_centers, bins = quantize_directions([0], n_dirs)
    vectors = np.stack([np.cos(bin_centers), np.sin(bin_centers)], 0)
    reg_vectors = np.reshape(
        vectors @ np.reshape(reg[:, :, :], (n_dirs, -1)),
        (2,) + reg.shape[1:],
    )

    # Extract angle (preferred direction) and amplitude (tuning strength)
    angle = np.arctan2(reg_vectors[1], reg_vectors[0])
    amp = np.sqrt(np.sum(reg_vectors ** 2, 0))

    return amp, angle


def color_stack_3d(
        amp,
        angle,
        hueshift=2.5,
        amp_percentile=80,
        maxsat=50,
        lightness_min=100,
        lightness_delta=-40,
    ):
    """
        Generates a colored representation of directional tuning data. Maps tuning properties to colors: angle
        (direction) → hue, amplitude → saturation.
    """

    # Normalize amplitude values based on percentile
    output_lch = np.empty(amp.shape + (3,))
    maxamp = np.percentile(amp, amp_percentile)

    # Map amplitude to lightness (stronger signals = darker)
    output_lch[:, :, 0] = (
            lightness_min + (np.clip(amp / maxamp, 0, 1)) * lightness_delta
    )
    output_lch[:, :, 1] = (np.clip(amp / maxamp, 0, 1)) * maxsat
    output_lch[:, :, 2] = (-angle + hueshift) * 180 / np.pi

    return JCh_to_RGB255(output_lch)


