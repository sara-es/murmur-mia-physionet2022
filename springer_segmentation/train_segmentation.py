import sys

from sklearn.linear_model import LogisticRegression
from tqdm import tqdm, trange
from tqdm.contrib import tzip
import os
import numpy as np
import scipy.io

from springer_segmentation.extract_features import getSpringerPCGFeatures
from springer_segmentation.run_segmentation import run_hmm_segmentation
from springer_segmentation.utils import get_wavs_and_tsvs


def train_hmm_segmentation(recordings, annotations, recording_freq=4000, feature_freq=50, use_psd=True):
    """

    Parameters
    ----------
    recordings
    annotations
    recording_freq

    Returns
    -------

    """

    # recordings is a list of recording
    number_of_states = 4
    numPCGs = len(recordings)
    # state_observation_values = np.zeros((numPCGs, number_of_states))
    state_observation_values = []

    for rec_idx in trange(len(recordings)):
        full_recording = recordings[rec_idx]

        if annotations[rec_idx].shape[0] == 3 and annotations[rec_idx].shape[1] != 3: # hacky workaround to hackier data handling
            annotation = annotations[rec_idx].T
        else:
            annotation = annotations[rec_idx]

        if annotation.shape[0] <= 1:
            continue

        clipped_recordings, segmentations = create_segmentation_array(full_recording,
                                                                    annotation,
                                                                    recording_frequency=recording_freq,
                                                                    feature_frequency=feature_freq)

        for clipped_recording, segmentation in zip(clipped_recordings, segmentations):
            PCG_Features, featuresFs = getSpringerPCGFeatures(clipped_recording,
                                                              recording_freq,
                                                              use_psd=use_psd,
                                                              featureFs=feature_freq)

            these_state_observations = []
            for state_i in range(1, number_of_states + 1):
                if PCG_Features.shape[0] != segmentation.shape[0]:
                    min_length = min(PCG_Features.shape[0], segmentation.shape[0])
                    PCG_Features = PCG_Features[:min_length]
                    segmentation = segmentation[:min_length]
                these_state_observations.append(PCG_Features[segmentation == state_i, :])
            state_observation_values.append(these_state_observations)

    models, pi_vector, total_obs_distribution = _fit_model(state_observation_values)

    return models, pi_vector, total_obs_distribution


def get_recordings_and_segmentations():
    segmentations = []
    clipped_recordings = []
    test_segmentations = []
    input_folder = sys.argv[1]

    for file_ in tqdm(sorted(os.listdir(input_folder))):
        root, extension = os.path.splitext(file_)
        if extension == ".wav":
            segmentation_file = os.path.join(input_folder, root + ".tsv")
            if not os.path.exists(segmentation_file):
                continue
            frequency, recording = scipy.io.wavfile.read(os.path.join(input_folder, file_))
            tsv_segmentation = np.loadtxt(segmentation_file, delimiter="\t")

            clipped_recording, test_segmentation = create_segmentation_array(recording,
                                                             tsv_segmentation,
                                                             recording_frequency=frequency,
                                                             feature_frequency=frequency)
            segmentations.append(tsv_segmentation)
            test_segmentations.append(test_segmentation)
            clipped_recordings.append(clipped_recording.reshape(-1))

    return clipped_recordings, segmentations, test_segmentations

def get_full_recordings(input_folder):
    recordings = []
    names = []

    for file_ in tqdm(sorted(os.listdir(input_folder))):
        root, extension = os.path.splitext(file_)
        if extension == ".wav":
            segmentation_file = os.path.join(input_folder, root + ".tsv")
            if not os.path.exists(segmentation_file):
                continue
            frequency, recording = scipy.io.wavfile.read(os.path.join(input_folder, file_))
            recordings.append(recording)
            names.append(file_)
    return recordings, names


def create_segmentation_array(recording,
                              tsv_segmentation,
                              recording_frequency,
                              feature_frequency=50):
    """

    Parameters
    ----------
    recording
    tsv_segmentation
    recording_frequency : int
        Frequency at which the recording is sampled
    feature_frequency : int
        Frequency of the features extracted in order to train the segmentation. The default, 50, is
        the frequency used in the matlab implementation

    Returns
    -------

    """

    full_segmentation_array = np.zeros(int(recording.shape[0] * feature_frequency / recording_frequency))

    for row_idx in range(0, tsv_segmentation.shape[0]):
        row = tsv_segmentation[row_idx, :]
        start_point = np.round(row[0] * feature_frequency).astype(int)
        end_point = np.round(row[1] * feature_frequency).astype(int)
        full_segmentation_array[start_point:end_point] = int(row[2])

    start_indices = []
    end_indices = []
    segmentations = []
    segment_started = False
    TOLERANCE = 5
    for idx in range(full_segmentation_array.shape[0]):
        if not segment_started and full_segmentation_array[idx] == 0:
            continue
        if full_segmentation_array[idx] != 0:
            if not segment_started:
                start_indices.append(idx)
                segment_started = True
                tol_counter = 0
            if tol_counter > 0:
                for adjust_idx in range(tol_counter):
                    full_segmentation_array[idx - adjust_idx - 1] = full_segmentation_array[idx - tol_counter - 1]
                tol_counter = 0
        if segment_started and full_segmentation_array[idx] == 0:
            tol_counter += 1
        if tol_counter == TOLERANCE or idx == full_segmentation_array.shape[0] - 1:
            end_indices.append(idx - tol_counter)
            if end_indices[-1] - start_indices[-1] > feature_frequency:
                segmentations.append(full_segmentation_array[start_indices[-1]:end_indices[-1]].astype(int))
            else:
                end_indices = end_indices[:-1]
                start_indices = start_indices[:-1]
            segment_started = False

    clipped_recordings = []
    for start, end in zip(start_indices, end_indices):
        clip = recording[int(start * recording_frequency / feature_frequency):int(end * recording_frequency / feature_frequency)]
        clipped_recordings.append(clip)

    # segmentation_array = segmentation_array[seg_start:seg_end].astype(int)
    return clipped_recordings, segmentations


def _fit_model(state_observation_values):

    number_of_states = 4
    pi_vector = 0.25 * np.ones(4)
    num_features = state_observation_values[0][0].shape[1]

    models = []
    statei_values = [np.zeros((0, num_features)) for _ in range(number_of_states)]

    for PCGi in range(len(state_observation_values)):
        for idx in range(4):
            statei_values[idx] = np.concatenate((statei_values[idx], state_observation_values[PCGi][idx]), axis=0)

    total_observation_sequence = np.concatenate(statei_values, axis=0)
    total_obs_distribution = []
    total_obs_distribution.append(np.mean(total_observation_sequence, axis=0))
    total_obs_distribution.append(np.cov(total_observation_sequence.T))

    for state_idx in range(number_of_states):

        length_of_state_samples = statei_values[state_idx].shape[0]

        length_per_other_state = np.floor(length_of_state_samples / (number_of_states - 1))

        min_length_other_class = np.inf

        for other_state_idx in range(number_of_states):
            samples_in_other_state = statei_values[other_state_idx].shape[0]

            if not other_state_idx != state_idx:
                min_length_other_class = min(min_length_other_class, samples_in_other_state)

        if length_per_other_state > min_length_other_class:
            length_per_other_state = min_length_other_class

        training_data = [None, np.zeros((0, num_features))]

        for other_state_idx in range(number_of_states):
            samples_in_other_state = statei_values[other_state_idx].shape[0]

            if other_state_idx == state_idx:
                indices = np.random.permutation(samples_in_other_state)[:int(length_per_other_state * (number_of_states -1)) ]
                training_data[0] = statei_values[other_state_idx][indices, :]
            else:
                indices = np.random.permutation(samples_in_other_state)[:int(length_per_other_state) + 1]
                state_data = statei_values[other_state_idx][indices, :]
                training_data[1] = np.concatenate((training_data[1], state_data), axis=0)

        labels = 2 * np.ones(training_data[0].shape[0] + training_data[1].shape[0])
        labels[0:training_data[0].shape[0]] = 1

        all_data = np.concatenate(training_data, axis=0)

        regressor = LogisticRegression(multi_class="multinomial")
        regressor.fit(all_data, labels)
        models.append(regressor)

    # Might want to make B_matrix and actual ndarray rather than list of ndarrays
    # But for now, we also return the model, since it will be more useful than the matrix
    return models, pi_vector, total_obs_distribution

def main():
    import matplotlib.pyplot as plt
    recordings, segmentations, names = get_wavs_and_tsvs(return_names=True)
    ground_truth_segmentations = []
    clipped_recordings = []


    models, pi_vector, total_obs_distribution= train_hmm_segmentation(recordings[:10], segmentations[:10])
    idx = 0
    for rec, seg in zip(recordings[:100], segmentations[:100]):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                     seg,
                                                     recording_frequency=4000,
                                                     feature_frequency=4000)
        ground_truth_segmentations.append(ground_truth[0])
        clipped_recordings.append(clipped_recording[0])
    for rec, seg, name in tzip(clipped_recordings[:20], ground_truth_segmentations, names):
        annotation, hr = run_hmm_segmentation(rec,
                                              models,
                                              pi_vector,
                                              total_obs_distribution,
                                              use_psd=True,
                                              return_heart_rate=True)
        if True:
            plt.title(f"{name} {hr}")
            # plt.plot(np.array(np.arange(annotation.shape[0])/4000),  2 +  2 * clipped_recordings[idx]/clipped_recordings[idx].max(), label="recording", lw=0.2)
            plt.plot(np.array(np.arange(annotation.shape[0])/4000), seg, label="ground truth")
            plt.plot(np.array(np.arange(annotation.shape[0])/4000), annotation, label="model")
            plt.legend()
            plt.show()
        idx += 1


if __name__ == "__main__":
    main()

