import numpy as np
import pywt
# import matlab.engine

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:23:29 2022
@author: g51388dw
inputs:
    X: the original signal
    N: decomposition level
    Name: wavelet name to use

outputs:
    cD: N-row array containing the detail coefficients for up to N levels
    cA: N-row array containing approximation coefficients for up to N levels
"""


# code below for debugging
# eng = matlab.engine.start_matlab()
# ml_recording = eng.load("recording1.mat")
# recording = np.asarray(ml_recording["r"]).reshape(-1)
# recording = recording.copy()
# Name = 'rbio3.9'

def getDWT(X, N, Name):
    coeffs = pywt.wavedec(X, Name, mode='symmetric', level=N)
    length = len(X)

    # get the detail coefficients
    cD = np.zeros((N, length));
    for i in range(N):
        d = np.asarray(coeffs[N - i])
        d = np.repeat(d, 2 ** (i + 1))
        cD[i, :] = centered_slice(d, length)

    # Space cD according to spacing of floating point numbers:
    cD[abs(cD) < np.sqrt(np.spacing(1))] = 0

    # get the approximation coefficients
    cA = np.zeros((N, length))
    for i in range(N):
        a = appcoef(coeffs, Name, i + 1)
        a = np.repeat(a, 2 ** (i + 1))
        cA[i, :] = centered_slice(a, length)

    cA[abs(cA) < np.sqrt(np.spacing(1))] = 0

    return cD, cA


def centered_slice(X, L):
    shape = len(X)

    # calculate start and end indices for each axis
    starts = (shape - L) // 2
    stops = starts + L

    return X[starts:stops]


def appcoef(coeffs, wavelet, level, **kwargs):
    max_level = len(coeffs) - 1
    if level == max_level:
        return coeffs[0]
    # this function also calculates the IDWT (kinda confusing API)
    approx = pywt.waverec(coeffs[:-level], wavelet, **kwargs)
    # not sure why PyWavelets sometimes gives duplicate final rows/columns
    if np.abs(approx[-1] - approx[-2]) < 0.00001:
        approx = approx[:-1]
    return approx