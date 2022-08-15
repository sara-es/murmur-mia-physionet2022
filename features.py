import numpy as np
import python_speech_features
import scipy.signal
from scipy.special import xlogy
from scipy.stats import skew
import scipy
from scipy.fft import fft, fftfreq
from helper_code import *
from pywt import wavedec



def _binning(array, nbins=50):
    """
    Helper function which allows computation of entropy in the way done in Zabihi
    """
    hist, bin_edges = np.histogram(array, bins=nbins)
    bin_indices = np.digitize(array, bin_edges[1:-1])
    probabilities = hist[bin_indices]
    probabilities = probabilities / array.shape[0]
    return probabilities


def _calc_fft(recording):
    SAMPLE_RATE = 4000

    x = recording
    xlen = recording.shape[0]
    hw = scipy.signal.windows.hamming(xlen)
    # N = int(SAMPLE_RATE * xlen/1000) # sample_rate * (recording duration in seconds)
    x_ham = x * hw  # apply hamming window to recording

    yf = fft(x_ham)[:xlen // 2]  # take real positive half of fft
    xf = fftfreq(xlen, 1 / SAMPLE_RATE)[:xlen // 2]

    return yf, xf


def compute_time_domain_entropy(signal):
    """
    """
    signal = signal.astype("float")
    power = np.abs(signal) ** 2 / signal.shape[0]
    normalised_power = power / power.sum()
    entropy = - (xlogy(normalised_power, normalised_power)).sum()
    return entropy


def compute_freq_domain_entropy(signal):
    """
    """
    signal, _ = _calc_fft(signal)
    signal = signal[:1000]
    power = np.abs(signal)**2 / signal.shape[0]
    normalised_power = power / power.sum()
    entropy = - (normalised_power * np.log(normalised_power)).sum()
    return entropy


def entropy_features(recording, nbins=50):
    """
    Calculates the entropy and the tsallis entropy of the distribution of voltages in the frequency domain. Equations (1)
    and (2) from  ``Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016 1.0.0"

    Parameters
        recording (ndarray):
        **kwargs: optional arguments to be passed to python_speech_features.mfcc

    Returns
        cepstral_features (ndarray): numpy array of shape (3,) where the features are (3), (4) and (5) from
            ``Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016 1.0.0"

    """
    probabilities = _binning(recording, nbins=nbins)
    entropy = - np.sum(probabilities * np.log(probabilities))
    tsallis = (1. - np.sum(probabilities**2))
    entropy_features = np.array([entropy, tsallis])
    return entropy_features


def cepstral_features(recording, **kwargs):
    """
    Calculates cepstral features of the recording passed in as a numpy array

    Parameters
        recording (ndarray):
        **kwargs: optional arguments to be passed to python_speech_features.mfcc

    Returns
        cepstral_features (ndarray): numpy array of shape (3,) where the features are (3), (4) and (5) from
            ``Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016 1.0.0"

    """
    cepstral = python_speech_features.mfcc(recording, **kwargs)
    feature_1 = np.mean(np.min(cepstral, axis=1))
    mean_ceptrals = np.mean(cepstral, axis=1)
    feature_2 = np.mean((np.max(cepstral, axis=1) - mean_ceptrals)**2)
    feature_3 = np.mean((skew(cepstral, axis=1) - mean_ceptrals)**2)
    cepstral_features = np.array([feature_1, feature_2, feature_3])
    return cepstral_features


def wavelet_features(recording, nbins=50):
    """
    Calculates wavelet features of the recording passed as numpy array

    Parameters
        recording (ndarray):
        nbins (int): number of bins to be used in construcing the histogram

    Returns
        wavelet_features (ndarray): statistics of the approximation and detail coeffecients for the wavelet transform
            of the recording. Based on features (6), (7), (8), and (9) from
            ``Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016 1.0.0"

    """
    results = wavedec(recording, "db4", level=5)
    all_features = []
    for coeff_array in results:
        probs = _binning(coeff_array, nbins=nbins)
        entropy = - np.sum(probs * np.log(probs))
        renyi_entropy = -np.log(np.sum(probs**2))
        log_var = np.log2(np.var(coeff_array))
        all_features += [entropy, renyi_entropy, log_var]
    return np.array(all_features)


def power_centroid(recording):
    """
    I gave up on trying to figure out matlab code and am just doing something that seems vaguely similar. This should be
    Equation (10) from "Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016 1.0.0"

    Parameters
        recording (ndarray)

    Returns
        power_centroid (ndarray):
    """
    freqs, psd = scipy.signal.welch(recording)
    p_centroid = np.array([(freqs * psd**2).sum() / (psd ** 2).sum()])
    return p_centroid

  
def get_all_features(recording,
                     nbins=50,
                     use_entropy=True,
                     use_cepstral=True,
                     use_wavelet=True,
                     use_power_centroid=True):
    """
    Collect all features from functions in this file and store them all in a single 1-dimensional array

    Parameters
        recording (ndarray)

    Returns
        power_centroid (ndarray):
    """
    all_features = []
    if use_entropy:
        # 4 features
        all_features.append(np.array([compute_time_domain_entropy(recording), compute_freq_domain_entropy(recording)]))
        all_features.append(entropy_features(recording, nbins))
    if use_cepstral:
        # 3 features
        all_features.append(cepstral_features(recording))
    if use_wavelet:
        # 18 features
        all_features.append(wavelet_features(recording, nbins=nbins))
    if use_power_centroid:
        # 1 feature
        all_features.append(power_centroid(recording))
    # all_features.append(extract_freq_features_all_noepoch(recording)[0])
    # all_features.append(extract_freq_features_all_noepoch(recording)[1])
    all_features_np = np.concatenate(all_features)
    return all_features_np


def get_segmented_features(recording, label):
    """
    CURRENTLY NOT USED. Replaced by team_code.get_cardiac_cycle_features()
    
    Originally used to get features from annotated segmentations provided in training
    data. Note 'label' is the segmentation_indices. 

    labels (list): np.array of size 3xn, corresponding to start-index, end-index and 
            type of segmentations, where n is (number of state transitions) + 1 
    """
    # recording = patient_recordings[0][0]
    # recording_segs = recording_labels[0]
    seg_length = 1024 # zero pad each segment to the same length
    bands = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    label_types = [1,2,3,4]
    Hz = 4000
    
    ave_seg_features = np.zeros((len(label_types), len(bands)))
    ave_seg_length = [0, 0, 0, 0]
    std_seg_length = [0, 0, 0, 0]
    
    if label.size > 0:
        for idx, i in enumerate(label_types):
            # get index of recording_segs with label_type ==i
            seg_idx = np.where(label[2,:]==i)
            seg_idx = np.asarray(seg_idx).ravel()
            seg_features = np.zeros((len(seg_idx), len(bands)))
            
            times = []
            for j in range(len(seg_idx)):
                start_time = label[0, seg_idx[j]]
                end_time = label[1, seg_idx[j]]
                start_idx = round(start_time*Hz) # multiply the time by Hz to get the index
                end_idx = round(end_time*Hz)
                data_seg = recording[start_idx:end_idx]
                
                #while we're here, let's get the average time for each segment
                times.append(end_time-start_time)
                
                #apply hamming window to segment, and then pad to xxxx
                ham = np.hamming(len(data_seg)) # calculating window in loop each time because it might get truncated
                data_seg = np.multiply(data_seg,ham)
                data_seg = np.pad(data_seg, (0, seg_length - len(data_seg)%seg_length), 'constant')
            
                #take fft            
                seg_fft = np.abs(np.fft.rfft(data_seg))**2
                seg_power = sum(seg_fft)/2
            
                for k in range(len(bands)):
                    band = bands[k]
                    seg_features[j,k] = np.sum(seg_fft[band:(band+10)]) / seg_power
            
            ave_seg_features[i-1,:] = np.sum(seg_features,0)
            ave_seg_length[idx] = np.mean(times)
            std_seg_length[idx] = np.std(times)
    return ave_seg_features, ave_seg_length, std_seg_length


def main():
    patient_files = find_patient_files("tiny_test")
    n_patient_files = len(patient_files)
    data = []
    recordings = []
    for idx in range(n_patient_files):
        patient_data = load_patient_data(patient_files[idx])
        data.append(patient_data)
        patient_recordings, frequencies = load_recordings("tiny_test", patient_data, get_frequencies=True)

        recordings.append(patient_recordings)

    first_recording = recordings[2][0]
    print(first_recording.shape)
    print(frequencies)


if __name__ == "__main__":
    main()