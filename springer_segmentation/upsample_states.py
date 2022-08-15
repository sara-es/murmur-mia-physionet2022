import numpy as np

def upsample_states(original_qt, old_fs, new_fs, new_length):
    """

    Parameters
    ----------
    original_qt
       The states inferred from the recording features (sampled at old_fs)
    old_fs
        The sampling frequency of the features from which the states were derived
    new_fs
        The desired sampling frequency
    new_length
        The desired length of the new signal

    Returns
    -------
    expanded_qt
        the inferred states resampled to be at frequency new_fs

    """

    original_qt = original_qt.reshape(-1)
    expanded_qt = np.zeros(new_length)

    indices_of_changes =  np.nonzero(np.diff(original_qt))[0]
    indices_of_changes = np.concatenate((indices_of_changes, [original_qt.shape[0] - 1]))

    start_index = 0
    for idx in range(len(indices_of_changes)):
        end_index = indices_of_changes[idx]

        # because division by 2 only has 0.0 and 0.5 as fractional parts, we can use ceil instead of round to stay faithful to MATLAB
        #mid_point = int(np.ceil((end_index - start_index) / 2) + start_index)
        # We don't need value at midpoint, we can just use the value at the start_index

        # value_at_midpoint = original_qt[mid_point]
        value_at_midpoint = original_qt[end_index]
#        if start_index != 0:
#            assert original_qt[start_index + 1] == original_qt[end_index]

        expanded_start_index = int(np.round((start_index) / old_fs * new_fs)) + 1
        expanded_end_index = int(np.round((end_index) / old_fs * new_fs))

        if expanded_end_index > new_length:
            expanded_end_index = new_length
        if idx == len(indices_of_changes) - 1:
            expanded_end_index = new_length + 1

        expanded_qt[expanded_start_index - 1:expanded_end_index] = value_at_midpoint
        start_index = end_index


    # expanded_qt = expanded_qt.reshape(-1, 1)
    return expanded_qt
