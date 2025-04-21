import numpy as np


def exp_decay_kernel(tau, dt, len_rec):
    upsample = 10
    t = np.arange(len_rec * upsample) * dt / upsample

    decay = np.exp(-t / tau)
    decay /= np.sum(decay)
    return decay


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


def normalize_traces(traces):

    norm_traces = np.copy(traces)
    norm_traces = norm_traces.T # need to transpose it since the functions work like that
    sd = np.nanstd(norm_traces)
    mean = np.nanmean(norm_traces)
    norm_traces = norm_traces-mean #numerator in the formula for z-score
    norm_traces = norm_traces/sd
    norm_traces = norm_traces.T
    return norm_traces