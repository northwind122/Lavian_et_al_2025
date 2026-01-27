import numpy as np


def exp_decay_kernel(tau, dt, len_rec):
    """
    Creates an exponential decay kernel for temporal filtering of stimulus data.

    This function generates a normalized exponential decay function that can be
    used to convolve with stimulus data to generate regressors.

    Parameters
    ----------
    tau : float
        Time constant of exponential decay in same units as dt
    dt : float
        Time step of the recording in seconds
    len_rec : int
        Length of the resulting kernel in number of time steps

    Returns
    -------
    decay : ndarray
        Normalized exponential decay kernel
    """
    upsample = 10
    t = np.arange(len_rec * upsample) * dt / upsample

    decay = np.exp(-t / tau)
    decay /= np.sum(decay)
    return decay


def corr2_coeff(A, B):
    """
    Computes the Pearson correlation coefficient between each row of matrix A
    and each row of matrix B, returning a correlation matrix.

    Parameters
    ----------
    A : ndarray
        First input matrix of shape (n_rows_A, n_columns)
    B : ndarray
        Second input matrix of shape (n_rows_B, n_columns)

    Returns
    -------
    corr : ndarray
        Correlation matrix of shape (n_rows_A, n_rows_B) where each element [i,j]
        is the correlation coefficient between row i of A and row j of B

    """
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def normalize_traces(traces):
    """
    Z-scoring neural activity traces (fast implementation).

    Parameters
    ----------
    traces : ndarray
        Neural activity traces with shape (time, ROIs)

    Returns
    -------
    norm_traces : ndarray
        Z-scored traces with same shape as input, normalized to mean=0, std=1
    """
    norm_traces = np.copy(traces)
    norm_traces = norm_traces.T
    sd = np.nanstd(norm_traces)
    mean = np.nanmean(norm_traces)
    norm_traces = norm_traces-mean
    norm_traces = norm_traces/sd
    norm_traces = norm_traces.T

    return norm_traces