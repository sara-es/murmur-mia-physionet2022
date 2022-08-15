from .extract_features import get_butterworth_high_pass_filter, get_butterworth_low_pass_filter, \
    get_homomorphic_envelope_with_hilbert
from .schmidt_spike_removal import schmidt_spike_removal

import numpy as np
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt


def get_heart_rate(audio_data,
                   Fs,
                   min_heart_rate=60,
                   max_heart_rate=190,
                   multiple_rates=True):
    """

    Parameters
    ----------
    audio_data
    Fs

    Returns
    -------

    """

    audio_data = get_butterworth_low_pass_filter(audio_data, 2, 100, Fs)
    audio_data = get_butterworth_high_pass_filter(audio_data, 2, 20, Fs)

    audio_data = schmidt_spike_removal(audio_data, Fs)

    homomorphic_envelope = get_homomorphic_envelope_with_hilbert(audio_data, Fs)

    y = homomorphic_envelope - homomorphic_envelope.mean()

    c = correlate(y, y)
    c /= c[int(c.shape[0]/2)]

    # check size of this
    signal_autocorrelation = c[homomorphic_envelope.shape[0]:]
    # Since the max and min indices are determined by 1/heart_rate, the max and min are switched
    min_idx = 60 * Fs / max_heart_rate
    max_idx = 60 * Fs / min_heart_rate

    min_sys_duration = np.round(0.1 * Fs)

    if multiple_rates:

        cut_autocorr = signal_autocorrelation[int(min_idx):int(max_idx)]
        peaks, _ = find_peaks(cut_autocorr, width=200, prominence=0.1)

        true_indices = peaks + min_idx
        heart_rates = 60. / (true_indices / Fs)

        max_sys_durations = [np.round(((60. / hr) * Fs) / 2) for hr in heart_rates]

        """
        plt.plot(peaks + min_idx, cut_autocorr[peaks], "xr")
        plt.plot(range(int(min_idx), int(max_idx)), cut_autocorr)
        plt.axvline(min_idx, color="pink")
        plt.axvline(max_idx, color="pink")
        plt.show()
        """

        systolic_time_intervals = []
        for max_sd in max_sys_durations:
            pos = np.argmax(signal_autocorrelation[int(min_sys_duration): int(max_sd)])
            systolic_time_intervals.append( (min_sys_duration + pos) / Fs)

    if not multiple_rates or len(heart_rates) == 0:
        index = np.argmax(signal_autocorrelation[int(min_idx): int(max_idx)])

        true_idx = index + min_idx
        heart_rate = 60. / (true_idx / Fs)

        max_sys_duration = np.round(((60. / heart_rate) * Fs) / 2)
        pos = np.argmax(signal_autocorrelation[int(min_sys_duration): int(max_sys_duration)])
        systolic_time_interval = (min_sys_duration + pos) / Fs
        systolic_time_intervals = [systolic_time_interval]
        heart_rates = [heart_rate]


    return heart_rates, systolic_time_intervals
