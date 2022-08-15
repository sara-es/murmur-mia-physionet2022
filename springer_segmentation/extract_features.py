import librosa.util
import numpy as np
from scipy import signal as signal
from scipy.signal import butter, filtfilt, hilbert

from .schmidt_spike_removal import schmidt_spike_removal
from .getDWT import getDWT

# from scipy.signal import resample
from librosa import resample


def getSpringerPCGFeatures(audio_data,
                           Fs,
                           matlab_psd=False,
                           use_psd=True,
                           use_wavelet=True,
                           featureFs=50):
    """

    Parameters
    ----------
    audio_data
    Fs
    matlab_psd
    use_psd
    use_wavelet
    featureFs

    Returns
    -------

    """

    audio_data = get_butterworth_low_pass_filter(audio_data, 2, 400, Fs)
    audio_data = get_butterworth_high_pass_filter(audio_data, 2, 25, Fs)
    audio_data = schmidt_spike_removal(audio_data, Fs)

    all_features = []

    homomorphic_envelope = get_homomorphic_envelope_with_hilbert(audio_data, Fs)
    # downsampled_homomorphic_envelope = resample(homomorphic_envelope, int(np.round(homomorphic_envelope.shape[0] * featureFs /recording_frequency)))
    downsampled_homomorphic_envelope = resample(homomorphic_envelope, orig_sr=Fs, target_sr=featureFs)
    downsampled_homomorphic_envelope = normalise_signal(downsampled_homomorphic_envelope)
    all_features.append(downsampled_homomorphic_envelope)

    hilbert_envelope = get_hilbert_envelope(audio_data)
    # downsampled_hilbert_envelope = resample(hilbert_envelope, int(np.round(hilbert_envelope.shape[0] * featureFs /recording_frequency)))
    downsampled_hilbert_envelope = resample(hilbert_envelope, orig_sr=Fs, target_sr=featureFs)
    downsampled_hilbert_envelope = normalise_signal(downsampled_hilbert_envelope)
    all_features.append(downsampled_hilbert_envelope)

    if use_psd:
        psd = get_power_spectral_density(audio_data, Fs, 40, 60, use_matlab=matlab_psd)
        if not matlab_psd:
            psd = psd / 2
        psd = resample(psd,
                       orig_sr=(1+1e-9),
                       target_sr=downsampled_homomorphic_envelope.shape[0] / len(psd))
        # psd = librosa.util.fix_length(psd, size=downsampled_hilbert_envelope.shape[0], mode="edge")
        psd = normalise_signal(psd)
        all_features.append(psd)

    # wavelet features
    if use_wavelet:
        wavelet_level = 3
        wavelet_name = "rbio3.9"

        if len(audio_data) < Fs * 1.025:
            audio_data = np.concatenate((audio_data, np.zeros((round(0.025 * Fs)))))

        # audio needs to be longer than 1 second for getDWT to work
        cD, cA = getDWT(audio_data, wavelet_level, wavelet_name)

        wavelet_feature = abs(cD[wavelet_level - 1, :])
        wavelet_feature = wavelet_feature[:len(homomorphic_envelope)]

        downsampled_wavelet = resample(wavelet_feature, orig_sr=Fs, target_sr=featureFs)
        downsampled_wavelet = normalise_signal(downsampled_wavelet)
        all_features.append(downsampled_wavelet)

    features = np.stack(all_features, axis=-1)
    return features, featureFs


def get_butterworth_high_pass_filter(original_signal,
                                     order,
                                     cutoff,
                                     sampling_frequency):
    """

    Parameters
    ----------
    original_signal
    order
    cutoff
    sampling_frequency

    Returns
    -------

    """
    B_high, A_high = butter(order, 2 * cutoff / sampling_frequency, btype="highpass")
    high_pass_filtered_signal = filtfilt(B_high, A_high, original_signal, padlen=3*(max(len(B_high),len(A_high))-1))
    return high_pass_filtered_signal


def get_butterworth_low_pass_filter(original_signal,
                                    order,
                                    cutoff,
                                    sampling_frequency):
    """

    Parameters
    ----------
    original_signal
    order
    cutoff
    sampling_frequency

    Returns
    -------

    """
    B_low, A_low = butter(order, 2 * cutoff / sampling_frequency, btype="lowpass")

    # padlen made equivalent to matlabs using https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function
    low_pass_filtered_signal = filtfilt(B_low, A_low, original_signal, padlen=3*(max(len(B_low),len(A_low))-1))
    return low_pass_filtered_signal


def get_homomorphic_envelope_with_hilbert(input_signal, sampling_frequency, lpf_frequency=8):
    """

    Parameters
    ----------
    input_signal
    sampling_frequency
    lpf_frequency

    Returns
    -------

    """

    B_low, A_low = butter(1, 2 * lpf_frequency / sampling_frequency, btype="low")
    homomorphic_envelope = np.exp(filtfilt(B_low, A_low, np.log(np.abs(hilbert(input_signal))), padlen=3*(max(len(B_low),len(A_low))-1)))

    # Remove spurious spikes in first sample
    homomorphic_envelope[0] = homomorphic_envelope[1]

    return homomorphic_envelope


def get_hilbert_envelope(input_signal):
    """

    Parameters
    ----------
    input_signal

    Returns
    -------

    """

    hilbert_envelope = np.abs(hilbert(input_signal))

    return hilbert_envelope


def get_power_spectral_density(data, sampling_frequency, frequency_limit_low, frequency_limit_high, use_matlab=False):
    """

    Parameters
    ----------
    data
    sampling_frequency
    frequency_limit_low
    frequency_limit_high
    use_matlab

    Returns
    -------

    """
    # note that hamming window is implicit in the matlab function - this might be what was messing up the shapes
    if not use_matlab:
        f, t, Sxx = signal.spectrogram(data, sampling_frequency, window=('hamming'), nperseg=int(sampling_frequency / 41),
                                       noverlap=int(sampling_frequency / 81), nfft=sampling_frequency)
        # ignore the DC component - springer does this by returning freqs from 1 to round(sampling_frequency/2). We do the same by removing the first row.
        Sxx = Sxx[1:, :]

    else:
        f, t, Sxx = matlab_spectrogram(data, sampling_frequency)
        f = np.asarray(f)
        t = np.asarray(t)
        Sxx = np.asarray(Sxx)


    low_limit_position = np.where(f == frequency_limit_low)
    high_limit_position = np.where(f == frequency_limit_high)

    # Find the mean PSD over the frequency range of interest:
    # This indexing passes tests, but I don't know why
    psd = np.mean(Sxx[low_limit_position[0][0]:high_limit_position[0][0]+1, :], axis=0)

    return psd


def normalise_signal(signal):
    """

    Parameters
    ----------
    signal

    Returns
    -------

    """

    mean_of_signal = np.mean(signal)

    standard_deviation = np.std(signal)

    normalised_signal = (signal - mean_of_signal) / standard_deviation

    return normalised_signal

def matlab_spectrogram(data, sampling_frequency, eng=None):
    if eng is None:
        import matlab.engine
        eng = matlab.engine.start_matlab()
    eng.addpath("../Springer-Segmentation-Code/")
    result = eng.spectrogram(matlab.double(data),
                             matlab.double(sampling_frequency / 40.),
                             matlab.double(np.round(sampling_frequency/79.)),
                             matlab.double(np.arange(1, np.round(sampling_frequency/2) + 1)),
                             matlab.double(1000), nargout=4)
    return result[-3], result[-2], result[-1]