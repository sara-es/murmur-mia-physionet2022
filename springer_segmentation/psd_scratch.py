import matlab.engine
import numpy as np
import scipy.signal as signal
import scipy

import springer_segmentation.extract_features

print(scipy.__version__)

eng = matlab.engine.start_matlab()
eng.addpath("~/code/Springer-Segmentation-Code/")

ml_recording = eng.load("recording1.mat")
recording = np.asarray(ml_recording["r"]).reshape(-1)

from scipy.signal.windows import hamming

window = hamming(24, sym=False)

def get_PSD_feature_Springer_HMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high):
    # note that hamming window is implicit in the matlab function - this might be what was messing up the shapes
    f, t, Sxx = signal.spectrogram(data, sampling_frequency, window="hamming", nperseg=int(sampling_frequency / 40),
                                   noverlap=int(sampling_frequency / 80)+1, nfft=sampling_frequency)

    # ignore the DC component - springer does this by returning freqs from 1 to round(sampling_frequency/2). We do the same by removing the first row.
    Sxx = Sxx[1:, :]
    print(Sxx[:4, 0])

    # low_limit_position = np.where(f == frequency_limit_low)
    # high_limit_position = np.where(f == frequency_limit_high)
    low_limit_position = np.argmin(np.abs(f - frequency_limit_low))
    high_limit_position = np.argmin(np.abs(f - frequency_limit_high))

    print(high_limit_position)
    print(low_limit_position)
    # Find the mean PSD over the frequency range of interest:
    psd = np.mean(Sxx[low_limit_position:high_limit_position+1, :], axis=0)

    return psd


matlab_result = springer_segmentation.extract_features.get_power_spectral_density(ml_recording["r"], 1000., 50., 60.)
python_result = get_PSD_feature_Springer_HMM(recording, 1000, 50, 60)

ml_result = np.asarray(matlab_result).reshape(-1)

print(ml_result.shape)
print(python_result.shape)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2)
ax[0].plot(ml_result)
ax[1].plot(python_result)
plt.show()

print(ml_result / python_result)
print(ml_result[1:] / python_result[:-1])
print(ml_result[:-1] / python_result[1:])
