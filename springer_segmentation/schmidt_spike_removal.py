
import numpy as np

def schmidt_spike_removal(original_signal, fs):
    """
    
    % The spike removal process works as follows:
    % (1) The recording is divided into 500 ms windows.
    % (2) The maximum absolute amplitude (MAA) in each window is found.
    % (3) If at least one MAA exceeds three times the median value of the MAA's,
    % the following steps were carried out. If not continue to point 4.
    % (a) The window with the highest MAA was chosen.
    % (b) In the chosen window, the location of the MAA point was identified as the top of the noise spike.
    % (c) The beginning of the noise spike was defined as the last zero-crossing point before theMAA point.
    % (d) The end of the spike was defined as the first zero-crossing point after the maximum point.
    % (e) The defined noise spike was replaced by zeroes.
    % (f) Resume at step 2.
    % (4) Procedure completed.
    %
    
    Parameters
    ----------
    original_signal : nd_array of shape (recording_length,)
    fs : float
        Sampling Frequency

    Returns
    -------

    """

    windowsize = np.round(fs / 2).astype(int)
    trailingsamples = (original_signal.shape[0] % windowsize).astype(int)
    if trailingsamples == 0:
        sample_frames = np.reshape(original_signal, (windowsize, -1))
    else:
        sample_frames = np.reshape(original_signal[:-trailingsamples], (windowsize, -1))

    MAAs = np.max(np.abs(sample_frames))

    while np.any(MAAs > np.median(MAAs) * 3):

        # Which window has the max MAAs
        window_num = np.argmax(MAAs)
        val = MAAs[window_num, :]

        # What is the position of the spike in the window
        spike_position = np.argmax(np.abs(sample_frames[:, val]))

        # Find zero crossings
        zero_crossings = np.abs( np.diff(np.sign(sample_frames[:, window_num])))>1
        zero_crossings = np.append(zero_crossings, 0)

        pre_spike_crossings = np.where(zero_crossings[:spike_position] == 1)
        if pre_spike_crossings[0].shape[0] == 0:
            spike_start = 0
        else:
            spike_start = pre_spike_crossings[0][-1]

        post_spike_crossings = np.where(zero_crossings[spike_position:] == 1)
        if post_spike_crossings[0].shape[0] == 0:
            spike_end = zero_crossings.shape[0] - 1
        else:
            spike_end = post_spike_crossings[0][0]

        sample_frames[spike_start:spike_end, window_num] = 0.0001

        MAAs = np.max(np.abs(sample_frames))

    despiked_signal = np.reshape(sample_frames, -1)
    despiked_signal = np.append(despiked_signal, original_signal[despiked_signal.shape[0]:])

    return despiked_signal



