from matplotlib import pyplot as plt
from tqdm.contrib import tzip
import numpy as np

from springer_segmentation.run_segmentation import run_hmm_segmentation
from springer_segmentation.train_segmentation import train_hmm_segmentation, create_segmentation_array, \
    get_full_recordings
from springer_segmentation.utils import get_wavs_and_tsvs

def does_it_segment(data_folder, n_train=10):
    recordings, segmentations, names = get_wavs_and_tsvs(return_names=True)

    models, pi_vector, total_obs_distribution= train_hmm_segmentation(recordings[:n_train], segmentations[:n_train])

    all_recordings, names = get_full_recordings(data_folder)
    for recording, name in tzip(all_recordings, names):
        annotation, hr = run_hmm_segmentation(recording,
                                              models,
                                              pi_vector,
                                              total_obs_distribution,
                                              use_psd=True,
                                              return_heart_rate=True)
        plt.plot(annotation)
        plt.savefig(f"images/{name}.pdf")
        plt.clf()

def how_does_it_do():
    # Get recordings and segmentations
    recordings, segmentations, names = get_wavs_and_tsvs(return_names=True)
    ground_truth_segmentations = []
    clipped_recordings = []

    # Train HMM
    models, pi_vector, total_obs_distribution= train_hmm_segmentation(recordings, segmentations)

    # Get ground truth
    for rec, seg in zip(recordings[:500], segmentations[:500]):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                     seg,
                                                     recording_frequency=4000,
                                                     feature_frequency=4000)
        ground_truth_segmentations.append(ground_truth[0])
        clipped_recordings.append(clipped_recording[0])
    idx = 0
    accuracies = np.zeros(len(clipped_recordings[:500]))
    for rec, seg, name in tzip(clipped_recordings[:500], ground_truth_segmentations, names):
        print(name)
        annotation, hr = run_hmm_segmentation(rec,
                                              models,
                                              pi_vector,
                                              total_obs_distribution,
                                              use_psd=True,
                                              return_heart_rate=True,
                                              try_multiple_heart_rates=False)
        print(hr)
        accuracies[idx] = (seg == annotation).mean()
        idx += 1
    plt.hist(accuracies)
    print(f"average accuracy: {accuracies.mean()}")
    plt.show()


if __name__ == "__main__":
    how_does_it_do()
    # does_it_segment("tiny_test")

