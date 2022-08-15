"""
D. Springer, Logistic Regression-HSMM-based Heart Sound Segmentation
https://physionet.org/content/hss/1.0/
"""

from .upsample_states import upsample_states
from .heart_rate import get_heart_rate
from .extract_features import getSpringerPCGFeatures
from .viterbi import viterbi_decode_recording
import numpy as np

def run_hmm_segmentation(audio_data,
                         models,
                         pi_vector,
                         total_observation_distribution,
                         Fs = 4000,
                         min_heart_rate=60,
                         max_heart_rate=200,
                         use_wavelet=True,
                         use_psd=True,
                         try_multiple_heart_rates=False,
                         return_heart_rate=False):
    """
    Give segmentation of recording `audio_data` based on the Hidden Markov Model (HMM) defined by
    the parameters `models`, `pi_vector` and `total_observation_distribution`.

    Parameters
    ----------
    audio_data: ndarray of shape (recording_length,)
    Fs: float giving the sampling frequency
    pi_vector: ndarray of shape (num_states,)
        The array of initial state probabilities
    total_observation_distribution: list of ndarrays
        TODO
    models
    return_heart_rate : boolean (default=False)
        if True, the heart rate is returned along with the assigned states,
        if False, just the assigned states are returned

    Returns
    -------

    """

    PCG_features, featuresFs = getSpringerPCGFeatures(audio_data, Fs, use_psd=use_psd, use_wavelet=use_wavelet)

    heart_rates, systolic_time_intervals = get_heart_rate(audio_data,
                                                     Fs,
                                                     min_heart_rate=min_heart_rate,
                                                     max_heart_rate=max_heart_rate,
                                                     multiple_rates=try_multiple_heart_rates)

    best_delta = -np.inf
    for heart_rate, systolic_time_interval in zip(heart_rates, systolic_time_intervals):
        delta, _, qt = viterbi_decode_recording(PCG_features, pi_vector, models, total_observation_distribution, heart_rate,
                                            systolic_time_interval, featuresFs)

        if delta.sum() > best_delta:
            best_delta = delta.sum()
            best_heart_rate = heart_rate
            best_qt = qt

    assigned_states = upsample_states(best_qt, featuresFs, Fs, audio_data.shape[0])


    # print(f"heart rate: {best_heart_rate}")
    if return_heart_rate:
        return assigned_states, best_heart_rate
    return assigned_states
