import scipy
import scipy.io.wavfile
import numpy as np
import sys
import os
from tqdm import tqdm

from springer_segmentation.train_segmentation import create_segmentation_array


def main():
    """
    For each wav file in the input_folder with a corresponding tsv, clip the recording to the length of the non-zero
    annotations and save both the clipped recording and the annotation array to a .mat file in the output_folder.
    """
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if len(sys.argv)== 4:
        feature_frequency = sys.argv[3]
    else:
        feature_frequency = 50

    for file_ in tqdm(sorted(os.listdir(input_folder))):
        root, extension = os.path.splitext(file_)
        if extension == ".wav":
            segmentation_file = os.path.join(input_folder, root + ".tsv")
            if not os.path.exists(segmentation_file):
                continue
            frequency, recording = scipy.io.wavfile.read(os.path.join(input_folder, file_))
            tsv_segmentation = np.loadtxt(segmentation_file, delimiter="\t")

            clipped_recording, segmentation = create_segmentation_array(recording,
                                                                        tsv_segmentation,
                                                                        recording_frequency=frequency,
                                                                        feature_frequency=feature_frequency)

            matlab_dict = {"recording": clipped_recording,
                           "segmentation": segmentation.reshape(-1, 1)}
            scipy.io.savemat(os.path.join(output_folder, root + ".mat"), matlab_dict)

    print("done")
    return matlab_dict

if __name__ == "__main__":
    main()