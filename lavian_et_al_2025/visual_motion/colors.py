import numpy as np
import colorspacious

N_DIRS = 8
HUESHIFT = 2.5


def JCh_to_RGB255(x):
    output = np.clip(colorspacious.cspace_convert(x, "JCh", "sRGB1"), 0, 1)
    return (output * 255).astype(np.uint8)


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


def get_timing_from_current_phase(stimulus_log, with_pause=True):
    '''
    :param stimulus_log:
    :param with_pause:
    :return:
    '''

    t_stim = np.asarray(stimulus_log.t)
    phase_ind = np.where(stimulus_log.columns.str.endswith('current_phase'))[0][0]
    phase = np.diff(np.asarray(stimulus_log.iloc[:, phase_ind]))
    t_change_ind = np.where(phase > 0)[0]
    if t_change_ind[0] != 0:
        t_change_ind = np.insert(t_change_ind, 0, 0)
    t_change_ind = np.append(t_change_ind, len(t_stim) - 1)
    t_change = t_stim[t_change_ind]

    if with_pause:
        t_ind = np.reshape(t_change_ind[1:], (-1, 2))
        t = np.reshape(t_change[1:], (-1, 2))

    else:
        t_ind = np.zeros((np.shape(t_change)[0] - 1, 2))
        t_ind[:, 0] = t_change_ind[0:-1]
        t_ind[:, 1] = t_change_ind[1:]

        t = np.zeros((np.shape(t_change)[0] - 1, 2))
        t[:, 0] = t_change[0:-1]
        t[:, 1] = t_change[1:]
    t_ind = t_ind.astype(int)

    return t, t_ind, phase_ind


stimulus_coloring_mapping = {"E0040_motions_cardinal": stimulus_plot_e0040}