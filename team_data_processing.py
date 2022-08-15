import random
import numpy as np
import scipy.signal as signal
from sklearn.experimental import enable_iterative_imputer # do not remove - enables next line for sklearn < 0.21
from sklearn.impute import IterativeImputer
import team_helper_code, team_constants
from springer_segmentation import train_segmentation
from springer_segmentation import run_segmentation


def split_into_epochs(pat_ids, recordings, data_features, cc_features, labels, epoch_len=20000):
    """
    Splits recordings into smaller epochs. Can do preprocessing of recordings here, but note that this is called before get_recording_features. 

    Parameters
        recordings (arr): a single recording.
        epoch_len (int): desired length of epochs in s * 4000 (4kHz recordings)

    Returns
        recording_epochs (arr): the original recording split into 5s epochs of recording data 
    """
    all_epochs = []
    epoch_pat_ids = []
    epoch_data_features = []
    # epoch_recording_features = []
    epoch_cc_features = []
    epoch_labels = []

    for i, rec in enumerate(recordings):
        if rec.shape[0] >= epoch_len: # Basically every case (min size in train is 20608)
            n_epochs = rec.shape[0]//epoch_len
            clip_len = epoch_len*n_epochs
            
            # Randomly choose starting point based on number of splits
            start = random.randint(0, rec.shape[0] - clip_len)
            rec = rec[start:start + clip_len]
            
            # Split remaining recording into equal length arrays
            epochs = np.split(rec, n_epochs)
            epochs = np.vstack(epochs)
            
        else: # If there's not enough recording data, pad with zeros
            left = random.randint(0, epoch_len - rec.shape[0])
            right = epoch_len - rec.shape[0] - left
            zeros_padding_r = np.zeros(shape=(left), dtype=np.float32)
            zeros_padding_l = np.zeros(shape=(right), dtype=np.float32)
            epochs = np.hstack((zeros_padding_r, rec, zeros_padding_l)).astype(np.float32)
            epochs = np.reshape(epochs, (1, -1))

        all_epochs.append(epochs)
        
        # Duplicate to have same len list as recordings
        epoch_pat_ids.extend(team_helper_code.like_length(pat_ids[i], epochs))
        epoch_data_features.extend(team_helper_code.like_length(data_features[i], epochs))
        # epoch_recording_features.extend(team_helper_code.like_length(recording_features[i], epochs))
        epoch_cc_features.extend(team_helper_code.like_length(cc_features[i], epochs))
        epoch_labels.extend(team_helper_code.like_length(labels[i], epochs)) 

    all_epochs = np.vstack(all_epochs).astype(np.float32)
    epoch_pat_ids = np.vstack(epoch_pat_ids).astype(np.float32)
    epoch_data_features = np.vstack(epoch_data_features).astype(np.float32)
    # epoch_recording_features = np.vstack(epoch_recording_features).astype(np.float32)
    epoch_cc_features = np.vstack(epoch_cc_features).astype(np.float32)
    epoch_labels = np.vstack(epoch_labels).astype(np.float32)
        
    return epoch_pat_ids, all_epochs, epoch_data_features, epoch_cc_features, epoch_labels


def standardize_recording_length(recordings, rec_len=80000):
    """
    Alternative to split_into_epochs, if we're not using those.  

    Parameters
        recordings (arr): the complete array of recordings
        epoch_len (int): desired length of recordings in s * 4000 (4kHz recordings)

    Returns
        recs (arr): all recordings forced into rec_len 
    """
    n_recordings = len(recordings)
    recs = np.zeros((n_recordings, rec_len))
    
    for i, rec in enumerate(recordings):
        if rec.shape[0] >= rec_len: # (min size in train is 20608)
            # Randomly choose starting point based on number of splits
            start = random.randint(0, rec.shape[0] - rec_len)
            recs[i,:] = rec[start:start + rec_len]
            
        else: # If there's not enough recording data, pad with zeros
            left = random.randint(0, rec_len - rec.shape[0])
            right = rec_len - rec.shape[0] - left
            zeros_padding_r = np.zeros(shape=(left), dtype=np.float32)
            zeros_padding_l = np.zeros(shape=(right), dtype=np.float32)
            recs[i,:] = np.hstack((zeros_padding_r, rec, zeros_padding_l)).astype(np.float32)

    return recs.astype(np.float32)


def get_data_features(data):
    """
    Parameters
        data (str): patient data

    Returns
        features (arr): len-6 array of: [
            age (float), 
            female (bool), 
            male (bool), 
            height (float), 
            weight (float), 
            is_pregnant (bool) ]

    """
    # Extract the age group and replace with the (approximate) number of months for 
    # the middle of the age group.
    age_group = team_helper_code.get_age(data)

    if team_helper_code.compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif team_helper_code.compare_strings(age_group, 'Infant'):
        age = 6
    elif team_helper_code.compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif team_helper_code.compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif team_helper_code.compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = team_helper_code.get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if team_helper_code.compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif team_helper_code.compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = team_helper_code.get_height(data)
    weight = team_helper_code.get_weight(data)

    # Extract pregnancy status.
    is_pregnant = team_helper_code.get_pregnancy_status(data)

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant]))
    return np.asarray(features, dtype=np.float32)


def get_cardiac_cycle_features(recording, segmentation_indices, bands=np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])):
    """
    Extracts cardiac cycle segmentation and features from a single recording. Uses the Springer segmentation
    algorithm.

    Parameters
        recording (arr): recording audio array

    Returns
        cc_fts (arr): the average frequency domain features for each phase in the cardiac cycle, 
            or subsegments of those phases
        segmentation_indices (arr): the segmentation annotations for this recording
        
    """
    # these are arbitrary, but basically looked at a few spectra, bins<100 were where the power is, 
    # and I couldn't be bothered to convert bins->freqs
    # bands = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) 
    stages = [1,2,3,4] # corresponding to stages in cardiac cycle
    
    ave_seg_features = np.zeros((len(stages), len(bands)))
    ave_seg_length = np.zeros(4)
    std_seg_length = np.zeros(4)

    if segmentation_indices.size > 0:
        with np.errstate(all='raise'):
            ave_seg_features, ave_seg_length, std_seg_length = \
                get_cc_subseg_features(stages, segmentation_indices, recording, bands)

        # # TODO: add some additional features here: 
              
    # frequency band powers for portions of a segment
    cc_fts = np.hstack((ave_seg_features.ravel(), ave_seg_length, std_seg_length))
    
    # make sure none of the features are nans by setting any nans to zero
    cc_fts = np.nan_to_num(cc_fts)
    return cc_fts


def get_cc_subseg_features(stages, segmentation_indices, recording, bands):
    """
    Needs cleaning up - moved subseg fts here so I can toggle for testing (take a while to run)
    """
    seg_length = 1024 # zero pad each segment to the same length
    # bands = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) 
    stages = [1,2,3,4] # corresponding to stages in cardiac cycle
    Hz = 4000
    n_subsegments = 5
    # phase_subsegments = np.array([7, 7, 4, 4]) # number of subsegments that each segment will be split into

    ave_subseg_features = np.zeros((len(stages), n_subsegments, len(bands)))
    ave_seg_length = np.zeros(4)
    std_seg_length = np.zeros(4)

    for i in range(len(stages)):
        # get indices of segmentation_indices with label_type from {1,2,3,4}
        stage_segs = np.where(segmentation_indices[2,:] == stages[i])[0]
        subseg_features = np.zeros((len(stage_segs), n_subsegments, len(bands)))
        times = []

        for j in range(len(stage_segs)):
            start_idx = int(segmentation_indices[0, stage_segs[j]])
            end_idx = int(segmentation_indices[1, stage_segs[j]])
            rec_seg = recording[start_idx:end_idx+1] # add 1 as python does not include end index
            
            if rec_seg.shape[0] == 0: # in case we're using bad indices
                break        

            # while we're here, let's get the average time for each segment
            times.append((end_idx-start_idx)/Hz) 
            
            # ---let's go a level deeper and get frequency bands corresponding to parts of a sub-segment---#
            # split segment into n_subsegments
            subseg_idx = np.round(np.linspace(start_idx, end_idx+1, n_subsegments+1)) #n_subsegments + 1 because of fences vs fenceposts
            subseg_idx = subseg_idx.astype(int)

            for subs in range(n_subsegments):
                s_idx = subseg_idx[subs]
                e_idx = subseg_idx[subs + 1]
                rec_seg = recording[s_idx:e_idx]
                if rec_seg.shape[0] == 0: # in case we're using bad indices
                    break
                    
                ham = np.hamming(len(rec_seg))
                rec_seg = np.multiply(rec_seg,ham)
                rec_seg = np.pad(rec_seg, (0, seg_length - len(rec_seg)%seg_length), 'constant')
                seg_fft = np.abs(np.fft.rfft(rec_seg))**2
                seg_power = sum(seg_fft)/2
                
                for k in range(len(bands)-1):
                    band = bands[k]
                    band_end = bands[k+1]
                    subseg_features[j,subs,k] = np.sum(seg_fft[band:band_end]) / seg_power 
            
        if len(stage_segs) != 0:
            ave_subseg_features[i,:,:] = np.sum(subseg_features,0)/len(stage_segs) #divide by length of epoch, basically
            
        ave_seg_length[i] = np.nanmean(times)
        std_seg_length[i] = np.nanstd(times)

    return ave_subseg_features, ave_seg_length, std_seg_length


def get_segmentation_indices(assigned_states):
    """
    Turns the array of assigned_states from the Springer segmentation into a np.array corresponding to
    start-time, end-time and type of segmentations.
    Similar format as is given in the challenge annotated .tsv files, except using sample index instead
    of time in seconds. Use idx/4000 if times, instead of indices, are required.

    Parameters
        assigned_states (arr): the array of labels containing {1,2,3,4} corresponding to phase in the 
            cardiac cycle

    Returns
        labels (arr): np.array of size 3xn, corresponding to start-index, end-index and 
            type of segmentations, where n is (number of state transitions) + 1
    """
    augmented = np.hstack((0, assigned_states))
    starts_idx = np.where(augmented[:-1] != augmented[1:])[0] # np.where returns a tuple
    stops_idx = np.hstack((starts_idx[1:] - 1, assigned_states.shape[0])) # append last index
    seg_types = assigned_states[starts_idx]

    return np.vstack((starts_idx, stops_idx, seg_types)).astype(int)


def train_impute_model(pat_ids, data_features):
    """
    Parameters
    ----------
    pat_ids : np array of size [num_epochs, 1]
    data_features : np array of size [num_epochs, 6]
        the 6 columns correspond to age(months), female, male, height, weight, is_pregnant

    Returns
    -------
    imp : fit imputation model (sklearn object)
    """
    #get indices of the unique patient ids
    u, indices = np.unique(pat_ids, return_index=True)
    patient_data_features = data_features[indices,:]

    # run the imputer on a per-patient level
    imp = IterativeImputer(max_iter=20)
    imp.fit(patient_data_features)
    
    return imp


def impute_data_features(imp_model, data_features):
    """
    Returns imputed version of data_features using MICE data imputation
    Imputation is on a *per-patient* basis, and the imputation transformation is applied to each row of data (epoch)
    
    Parameters
    ----------
    imp_model : fit imputation model
    data_features : np array of size [num_epochs, 6]
        the 6 columns correspond to age(months), female, male, height, weight, is_pregnant

    Returns
    -------
    imputed_data_features : np array of size == data_features

    """
    mins = team_constants.MINS
    maxs = team_constants.MAXS

    # apply imputer at a per-epoch level
    data_features = imp_model.transform(data_features)
    
    # reset any out of range imputed values to be between the minimum and maximum values in the dataset
    for i in range(len(mins)):
        data_features[data_features[:,i] < mins[i] , i] = mins[i]
        data_features[data_features[:,i] > maxs[i] , i] = maxs[i]
    
    return data_features


def get_envelope(arr, w):
    return np.convolve(np.abs(arr), signal.triang(w))/w*2

def take_epoch_envelope(epochs_arr, w=100):
    env_arr = []
    for arr in epochs_arr:
        env_arr.append(get_envelope(arr, w))
    return np.asarray(env_arr).astype(np.float32)

def define_hr_thresh(age_group, scale_thresh = 1):
    hr_lims = np.empty(2)
    if age_group == 0.5:
        hr_lims[0] = 80
        hr_lims[1] = 200
    elif age_group == 6:
        hr_lims[0] = 80
        hr_lims[1] = 190 
    elif age_group == 6 * 12:
        hr_lims[0] = 50
        hr_lims[1] = 140 
    elif age_group == 15 * 12:
        hr_lims[0] = 35
        hr_lims[1] = 120
    elif age_group == 20 * 12:
        hr_lims[0] = 30
        hr_lims[1] = 120
    else:
        #in case age is NaN
        hr_lims[0] = 30
        hr_lims[1] = 200
        
    # apply scaling
    hr_lims[0] = hr_lims[0] * 1/scale_thresh
    hr_lims[1] = hr_lims[1] * scale_thresh    
    
    return hr_lims


def load_preprocessed_segmentations(tsv_segmentations, hr_list=None):
    """
    In case we've pickled this already
    """
    Hz = 4000
    segmentations = []

    for patient_segs in tsv_segmentations: # given on a per-patient basis
        for i, seg in enumerate(patient_segs):
            if seg.size > 0:
                segmentation_indices = np.zeros_like(seg)
                # use given segmentations, but these are given in s, so convert to indices
                segmentation_indices[0,:] = np.round(seg[0]*Hz).astype(int)
                segmentation_indices[1,:] = np.round(seg[1]*Hz).astype(int)
                segmentation_indices[2,:] = seg[2].astype(int)
            else:
                segmentation_indices = np.zeros((3,1))
            
            # extracted_segmentations is now a flattened list
            segmentations.append(segmentation_indices)
    
    heart_rates = []
    if hr_list is not None:
        for patient_hrs in hr_list:
            for hr in patient_hrs:
                heart_rates.append(hr)

    return segmentations, heart_rates

