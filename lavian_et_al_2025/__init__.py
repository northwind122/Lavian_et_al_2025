import numpy as np


def exp_decay_kernel(tau, dt, len_rec):
    upsample = 10
    t = np.arange(len_rec * upsample) * dt / upsample

    decay = np.exp(-t / tau)
    decay /= np.sum(decay)
    return decay


def corr2_coeff(A, B):
    # Row-wise mean of input arrays & subtract from input arrays 
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

