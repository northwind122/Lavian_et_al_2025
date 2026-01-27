import numpy as np
import colorspacious

N_DIRS = 8
HUESHIFT = 2.5


def JCh_to_RGB255(x):
    """
    Converts colors from JCh color space to RGB for display.

    Parameters
    ----------
    x : ndarray
        Array of shape (..., 3) containing JCh values

    Returns
    -------
    ndarray
        Array of shape (..., 3) containing RGB values (0-255)
    """
    output = np.clip(colorspacious.cspace_convert(x, "JCh", "sRGB1"), 0, 1)
    return (output * 255).astype(np.uint8)


def get_paint_function(stimulus_log, protocol_name):
    """
    Generates color coding for the visual motion experiment stimuli.

    Colors are assigned based on motion direction, with each direction mapped
    to a specific hue. Stationary periods are colored black.

    Parameters
    ----------
    stimulus_log : DataFrame
        The stimulus log containing background motion parameters

    Returns
    -------
    s : ndarray
        RGB color values for each stimulus phase
    t_change : ndarray
        Timing information for stimulus phase changes
    """
    # Extract timing information and indices of phase changes
    t_change, t_change_ind = get_timing_from_current_phase(stimulus_log)[0:2]

    # Calculate motion displacement between frames
    sx = np.diff(np.asarray(stimulus_log.bg_x))[t_change_ind[:, 0]]
    sy = np.diff(np.asarray(stimulus_log.bg_y))[t_change_ind[:, 0]]

    # Identify stationary periods (no motion)
    ind0 = np.where((sx == 0) & (sy == 0))[0]

    # Calculate motion direction angles
    angles = np.arctan2(sy, sx)
    angles = np.stack(
        [
            np.full(len(sx), 60),
            np.full(len(sx), 60),
            (-angles + 2.5) * 180 / np.pi,
        ],
        1)

    # Convert to RGB colors
    s = JCh_to_RGB255(angles)
    s[ind0, :] = 0

    return s, t_change


def get_timing_from_current_phase(stimulus_log, with_pause=True):
    """
    Extracts stimulus phase timing information from the stimulus log.

    Identifies time points where the stimulus phase changes, which represent
    transitions between different stimulus conditions.

    Parameters
    ----------
    stimulus_log : DataFrame
        The stimulus log acquired using the Stytra software
    with_pause : bool, default=True
        If True, assumes alternating stimulus/pause structure

    Returns
    -------
    t : ndarray
        Time points for phase start and end
    t_ind : ndarray
        Indices of phase start and end in the stimulus log
    phase_ind : int
        Column index of the phase information in the stimulus log
    """
    # Extract time points from stimulus log
    t_stim = np.asarray(stimulus_log.t)
    # Find the column containing phase information
    phase_ind = np.where(stimulus_log.columns.str.endswith('current_phase'))[0][0]

    # Detect phase transitions
    phase = np.diff(np.asarray(stimulus_log.iloc[:, phase_ind]))
    t_change_ind = np.where(phase > 0)[0]
    if t_change_ind[0] != 0:
        t_change_ind = np.insert(t_change_ind, 0, 0)
    t_change_ind = np.append(t_change_ind, len(t_stim) - 1)
    t_change = t_stim[t_change_ind]

    if with_pause:
        # For protocols that alternate between stimulus and pause
        # group time points as (stimulus_start, pause_start)
        t_ind = np.reshape(t_change_ind[1:], (-1, 2))
        t = np.reshape(t_change[1:], (-1, 2))

    else:
        # For protocols with consecutive stimuli without pauses
        # group time points as (phase_start, phase_end)
        t_ind = np.zeros((np.shape(t_change)[0] - 1, 2))
        t_ind[:, 0] = t_change_ind[0:-1]
        t_ind[:, 1] = t_change_ind[1:]

        t = np.zeros((np.shape(t_change)[0] - 1, 2))
        t[:, 0] = t_change[0:-1]
        t[:, 1] = t_change[1:]
    t_ind = t_ind.astype(int)

    return t, t_ind, phase_ind

