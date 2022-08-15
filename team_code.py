#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you 
# can edit most parts of the required functions, change or remove non-required functions, 
# and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

import os, glob, pickle
import numpy as np
import pandas as pd
import torch
import team_helper_code, team_constants, team_data_processing, combine_models
import outlier_model, gradient_boosting_model, rubin_cnn_model, nn_ensemble_model
from springer_segmentation import train_segmentation, run_segmentation
import combine_models
import time

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the 
# arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # DO NOT CHANGE ARGUMENTS
    # Get raw recordings and features
    if verbose >= 1:
        print('Loading data and recordings...')
    patient_data, patient_recordings, tsv_annotations = load_data_from_folder(data_folder, load_segmentations=True)
    # recording_segmentation = load_recording_segmentation(data_folder, patient_data)
    
    # called in a separate function, so we can do custom train test splits if needed
    train_model(patient_data, patient_recordings, tsv_annotations, model_folder, verbose, given_segmentations=None)

    if verbose >= 1:
        print('Done.')


def train_model(patient_data, patient_recordings, tsv_annotations, model_folder, verbose, given_segmentations=None, given_hrs=None):
    """
    Main function call in train_challenge_model.

    Parameters:
        patient_data (list): list of all patient data strings, of len(n_patients)
        patient_recordings (list): list of lists containing all recordings per patient, of len(n_patients)
        tsv_annotations (list): list of lists containing cardiac cycle annotations for each recording, per patient, of len(n_patients)
        model_folder: path to folder in which model should be stored
        verbose (bool): how many printouts do you want?

    Optional, for speedy testing:
        given_segmentations: pre-processed segmentation data
        given_cc_fts: pre-processed frequency domain features

    Returns: None. Saves trained models to model_folder.
    """
    start = time.time() # because I am impatient

    # create model dictionary 
    team_model = {}

    # Extract features needed to train the model
    pat_ids, recordings, data_features, tsvs, murmur_targets, outcome_targets, num_recs = \
        process_data_and_recordings(patient_data, patient_recordings, tsv_annotations=tsv_annotations)
    
    if outcome_targets.max() == 0:
        raise ValueError('Missing outcome targets?')

    # Impute tabular data features using MICE
    if verbose >= 1:
        print('Imputing missing data...')
    team_model['imputation'] = team_data_processing.train_impute_model(pat_ids, data_features)
    data_features = team_data_processing.impute_data_features(team_model['imputation'], data_features)
    
    if verbose >= 2:
        t1 = time.time()

    if given_segmentations is None: # if we haven't pickled this already
         ###### Train HMM model for cardiac cycle segmentations ######
        if verbose >= 1:
            print('Training cardiac cycle detection model...')
        hmm_model = {}
        # train the model on expert annotations
        hmm_model['models'], hmm_model['pi_vector'], hmm_model['total_obs_distribution'] = \
            train_segmentation.train_hmm_segmentation(recordings, tsvs)
        team_model['segmentation'] = hmm_model

        if verbose >= 1:
            print('Performing cardiac cycle inference...')
        # data_features[:,0] is ages in months
        segmentations, heart_rates = get_recording_segmentations(recordings, 
                                                                data_features[:,0], 
                                                                hmm_model)
    else: 
        segmentations, heart_rates = team_data_processing.load_preprocessed_segmentations(given_segmentations, given_hrs)

    data_features[:,2] = np.array(heart_rates) # replace male (bool) with heart rates

    # Extract frequency domain features
    if verbose >= 1:
        print('Extracting frequency domain features...')
    cc_features = extract_recording_features(recordings, segmentations)

    
    if verbose >= 2:
        t2 = time.time()
        print(f'Done. Time to extract segmentations and features: {t2 - t1:.2f} seconds')

    # Train the models.
    #### Model to determine 'unknown' class (previously novelty detection, but now using supervised grad. boosting) ########
    if verbose >= 1:
        print('Training outlier detector...')
    team_model['outlier_detector_murmur'] = outlier_model.train_model(recordings, 
                                                                    murmur_targets, 
                                                                    num_recs, 
                                                                    data_features[:,2]) # data_features[:,2] is heart rates

    team_model['outlier_detector_outcome'] = outlier_model.train_model(recordings, 
                                                                    outcome_targets,
                                                                    num_recs, 
                                                                    data_features[:,2])
    
        
    ###### Gradient boosting ######
    if verbose >= 1:
        print('Training gradient boosting models...')
    team_model['scaler'], team_model['gb_murmur_classifier'] = \
       gradient_boosting_model.train_model(cc_features, data_features, murmur_targets)

    _, team_model['gb_outcome_classifier'] = \
        gradient_boosting_model.train_model(cc_features, data_features, outcome_targets)


    # ##### MFCC NN #####
    if verbose >= 1:
        print('Training base cnns...')
    
    for i in range(5):
        mod_name = 'rubin_cnn_' + str(i)
        if verbose >= 2:
            print(f"Training murmur CNN {(i+1)}/5.")
        team_model[mod_name] = \
            rubin_cnn_model.train_rubin_cnn(recordings, segmentations, data_features, 
                                            murmur_targets, verbose, num_epochs=7)
               
    if verbose >= 2:
        print(f'Time to train individual models: {time.time() - t2:.2f} seconds')
    
    if verbose >= 1:
        print('All individual CNNs trained. Training ensemble...')
    
    team_model['NN_ensemble'] = nn_ensemble_model.train_cnn_ensemble(team_model['rubin_cnn_0'], 
                                                                team_model['rubin_cnn_1'], 
                                                                team_model['rubin_cnn_2'], 
                                                                team_model['rubin_cnn_3'], 
                                                                team_model['rubin_cnn_4'], 
                                                                murmur_targets, 
                                                                recordings, 
                                                                segmentations, 
                                                                data_features, 
                                                                learning_rate=0.00001, num_epochs=6, 
                                                                verbose=verbose)    
    
    if verbose >= 1:
        print(f'Saving model to disk...')
    os.makedirs(model_folder, exist_ok=True)       
    for name, model in team_model.items():
        if isinstance(model, torch.nn.Module):
            save_model_torch(model, name, model_folder)
        else:
            save_model_pkl(model, name, model_folder)
        if verbose >= 2:
            print(f'{name} model saved.')
    if verbose >= 1:
        print(f'Done. Total time elapsed: {time.time() - start:.2f} seconds')


# Load your trained model. This function is *required*. You should edit this function 
# to add your code, but do *not* change the arguments of this function.
def load_challenge_model(model_folder, verbose, models=['outlier_detector_murmur',
                                                        'outlier_detector_outcome',
                                                        'imputation', 
                                                        'segmentation', 
                                                        'scaler', 
                                                        'gb_murmur_classifier', 
                                                        'gb_outcome_classifier', 
                                                        'rubin_cnn_0', 
                                                        'rubin_cnn_1', 
                                                        'rubin_cnn_2', 
                                                        'rubin_cnn_3', 
                                                        'rubin_cnn_4',
                                                        'NN_ensemble',
                                                        ]):   
    # DO NOT CHANGE ARGUMENTS 
    models_trained = models 
    team_model = {}

    if verbose >= 1:
        print(f'Attempting to load from {model_folder}...')

    if os.path.exists(model_folder):
        for name in models_trained:
            # try to load pkl
            fnp = os.path.join(model_folder, name + '.sav')
            fnt = os.path.join(model_folder, name + '.pth')
            if os.path.exists(fnp):
                try:
                    model = pickle.load(open(fnp, 'rb'))
                    team_model[name] = model
                    if verbose >= 1:
                        print(f"Loaded {name} model.")
                except ValueError:
                    print(f"I couldn't load the pickled model {fnp}.")
            # else try to load torch
            elif os.path.exists(fnt):
                device = torch.device("cpu")
                try:
                    model = torch.load(fnt, map_location=device)
                    team_model[name] = model
                    if verbose >= 1:
                        print(f"Loaded {name} model.")
                except ValueError:
                    print(f"I couldn't load the torch model {fnt}.")
            else: print(f"I can't find the model {name}.")
    else: print(f"{model_folder} not found or is not a valid path.")
    
    return team_model


# Run your trained model. This function is *required*. You should edit this function 
# to add your code, but do *not* change the arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    # DO NOT CHANGE ARGUMENTS
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']
    
    outlier_probs_murmur, gb_probs_murmur, cnn_probs_murmur, \
         outlier_probs_outcome, gb_probs_outcome =\
            run_model(model, data, recordings, verbose)

    # combine cnn_probs_outcome with gb_probs_outcome using logistic regression
    pred_labels_murmur, all_probs_murmur = combine_models.combine_CNN_GB(cnn_probs_murmur, gb_probs_murmur)

    # Combine predictions from different models
    murmur_pred, murmur_probs = \
        combine_models.recording_to_murmur_predictions(outlier_probs_murmur, all_probs_murmur)
    outcome_pred, outcome_probs = \
        combine_models.recording_to_outcome_predictions(outlier_probs_outcome, gb_probs_outcome)
    # TODO there might be a problem with the gb_probs_outcome. index 0 is always 0. 
    # it shouldn't make a difference, but check this
    
    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    predictions = np.concatenate((murmur_pred, outcome_pred))
    probabilities = np.concatenate((murmur_probs, outcome_probs))

    return classes, predictions, probabilities


def run_model(model, data, recordings, verbose, given_segmentations=None, given_hrs=None):
    """
    Main function call in run_challenge_model.

    Parameters:
        model (dict): all saved models; see load_challenge_model for keys
        patient_data (str): a patient demographic data string
        patient_recordings (list): all recordings for one patient
        verbose (bool): how many printouts do you want?

    Returns: 
        outlier_probs_murmur
        gb_probs_murmur
        cnn_probs_murmur
        outlier_probs_outcome 
        gb_probs_outcome 
        murmur_targets 
        outcome_targets
    """
    patient_data = [data]
    patient_recordings = [recordings]

    # Extract features needed to run the model
    pat_ids, recordings, data_features, tsvs, murmur_targets, outcome_targets, num_recs = \
        process_data_and_recordings(patient_data, patient_recordings)

    # Impute tabular data features using MICE
    data_features = team_data_processing.impute_data_features(model['imputation'], data_features)

    ###### cardiac cycle segmentations ######    
    # data_features[:,0] is ages in months
    if given_segmentations is None:
        segmentations, heart_rates = get_recording_segmentations(recordings, 
                                                                data_features[:,0], 
                                                                model['segmentation'])
    else:
        segmentations, heart_rates = team_data_processing.load_preprocessed_segmentations([given_segmentations], [given_hrs])
    data_features[:,2] = np.array(heart_rates) 

    # Extract frequency domain features
    cc_features = extract_recording_features(recordings, segmentations)

    # Run models
    #### Unknown detection
    # returns outlier predictions per recording, outlier probabilities per recording
    _, outlier_probs_murmur = outlier_model.run_model(model['outlier_detector_murmur'], 
                                                    recordings, 
                                                    num_recs, 
                                                    data_features[:,2])

    _, outlier_probs_outcome = \
        outlier_model.run_model(model['outlier_detector_outcome'], 
                                    recordings,
                                    num_recs, 
                                    data_features[:,2])

    #### Gradient boosting on cardiac cycle features
    # gradient_boosting_model returns one-hot predictions per recording, probabilities per recording
    _, gb_probs_murmur = \
            gradient_boosting_model.run_model(model['scaler'], 
                                            model['gb_murmur_classifier'], 
                                            cc_features, 
                                            data_features, 
                                            )

    #### Ensembled CNNs
    # rubin_cnn_model returns one-hot predictions per recording, probabilities per recording
    _, cnn_probs_murmur = \
            nn_ensemble_model.run_nn_rubin_ensemble(model['NN_ensemble'],
                                        model['rubin_cnn_0'],
                                        model['rubin_cnn_1'],
                                        model['rubin_cnn_2'],
                                        model['rubin_cnn_3'],
                                        model['rubin_cnn_4'],
                                        recordings,
                                        segmentations,
                                        data_features)

    _, gb_probs_outcome = \
            gradient_boosting_model.run_model(model['scaler'], 
                                            model['gb_outcome_classifier'], 
                                            cc_features, 
                                            data_features, 
                                            )
    
    return outlier_probs_murmur, gb_probs_murmur, cnn_probs_murmur, \
         outlier_probs_outcome, gb_probs_outcome


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def save_model_pkl(model, name, folder):
    if model is not None:
        try:
            fn = os.path.join(folder, name + '.sav')
            pickle.dump(model, open(fn, 'wb'))
        except:
            print(f'Could not save model to {fn}.')

def save_model_torch(model, name, folder):
    if model is not None:
        try:
            fn = os.path.join(folder, name + '.pth')
            torch.save(model, fn)
        except:
            print(f'Could not save torch model to {fn}.')


def load_data_from_folder(data_folder, load_segmentations=False):
    """
    Load data and recordings from training folder. Returns data and recordings in the same 
    format that is given for run_model.

    Parameters
        data_folder: path to folder containing patient data and recordings

    Returns
        patient_data (list): list of all patient data strings, of len(n_patients)
        patient_recordings (list): list of lists containing all recordings per patient, 
            of len(n_patients)
    """

    # get files 
    patient_files = team_helper_code.find_patient_files(data_folder)
    n_patient_files = len(patient_files)
    patient_data = []
    patient_recordings = []
    segmentations = []

    for i in range(n_patient_files):
        p_data = team_helper_code.load_patient_data(patient_files[i])
        recs = team_helper_code.load_recordings(data_folder, p_data, get_frequencies=False)
        patient_data.append(p_data)
        patient_recordings.append(recs) # list of recordings arrays
        patient_segmentations = [] # make a list per patient, to mirror recordings list format

        if load_segmentations:
            num_locations = team_helper_code.get_num_locations(p_data)
            recording_information = p_data.split('\n')[1:num_locations+1] # gets the 4 files
            for i in range(num_locations):
                entries = recording_information[i].split(' ')
                try:
                    recording_file = entries[3] # this gets the tsv file - note that not all .wav files have a tsv. (e.g. patient 50782)
                    filename = os.path.join(data_folder, recording_file)
                    df=pd.read_csv(filename, sep='\t',header=None)
                    df = df.transpose()
                    df = df.to_numpy().astype(np.float32)
                    patient_segmentations.append(df)
                except ValueError: # can't access these in testing data?
                    break
                except:
                    patient_segmentations.append(np.empty([0,0]))
        segmentations.append(patient_segmentations)

    if load_segmentations:
        return patient_data, patient_recordings, segmentations

    return patient_data, patient_recordings, None


def process_data_and_recordings(patient_data, patient_recordings, tsv_annotations=None):
    """
    Takes raw data strings and list of recordings and returns arrays, all of equal length, of IDs, data features, and processed recordings.

    Parameters
        data (list): list of patient data strings
        recordings (list): list of lists of patient recordings
    
    Returns
        pat_ids (arr): integer patient IDs
        recordings (list): recording data. Last second or so has been removed.
        data_features (arr): [age, female, male, height, weight, is_pregnant] WARNING: contains NaNs
        murmur_labels (arr): murmur class target label: 0 if 'Present', 1 if 'Unknown', 2 if 'Absent'
        outcome_labels (arr): outcome class target label: 0 if 'Abnormal', 1 if 'Normal'
    """

    n_patient_files = len(patient_data)
    pat_ids = []
    data_features = []
    recordings = []
    annotations = []
    murmur_labels = []
    outcome_labels = []
    num_recs = []
    
    for i in range(n_patient_files):
        locations = team_helper_code.get_locations(patient_data[i])
        pat_id = team_helper_code.get_patient_id(patient_data[i]) 
        data_fts = team_data_processing.get_data_features(patient_data[i]) 
        
        try: # Labels may not be available in testing data 
            pat_label = team_helper_code.get_class_labels(patient_data[i])
            murmur_locations = team_helper_code.get_murmur_location(patient_data[i])
        except ValueError:
            pat_label = 0
            murmur_locations = []
        
        try: 
            pat_outcome = team_helper_code.get_outcome_labels(patient_data[i])
        except ValueError:
            # outcomes are in newer version of dataset
            # randint used for testing in order to have 2 labels, but should not be used in training
            pat_outcome = 0

        # Loop through available recordings and reassign labels on a per-recording basis as needed.
        for rec_idx, rec in enumerate(patient_recordings[i]):
            # If Murmur, but not audible at this location, set recording label to Absent
            if pat_label == 0 and locations[rec_idx] not in murmur_locations: 
                label = 2
            else: 
                label = pat_label
            murmur_labels.append(label) 

            cutoff = team_constants.CUTOFF_TIME
            rec = rec[:-cutoff] # remove the last second (stethoscope noise)
            
            recordings.append(rec)
            pat_ids.append(pat_id)
            data_features.append(data_fts)
            outcome_labels.append(pat_outcome)

            if tsv_annotations is not None: 
                annotations.append(tsv_annotations[i][rec_idx])
        
            #get number of recordings
            num_rec = len(patient_recordings[i])
            num_recs.append(num_rec)
        
    if len(annotations) != len(recordings) and tsv_annotations is not None:
        print('Are you missing an annotation file?')

    # Lists to numpy arrays (care, recordings and annotations are still lists)
    num_recs = np.vstack(num_recs).astype(np.float32)
    pat_ids = np.vstack(pat_ids).astype(np.float32)
    data_features = np.vstack(data_features).astype(np.float32)
    murmur_labels = np.vstack(murmur_labels).astype(np.float32)
    outcome_labels = np.vstack(outcome_labels).astype(np.float32)

    return pat_ids, recordings, data_features, annotations, murmur_labels, outcome_labels, num_recs


def get_recording_segmentations(recordings, ages, hmm_model, verbose=1):
    """
    Parameters
        recordings (list): list of individual recordings
        ages (arr): age of the patient corresponding to each recording, in months
        hmm_model (dict): dict containing models, pi_vector, total_obs_distribution

    Returns
        extracted_segmentations (list): lists of segmentations annotation arrays. Each array
            is a 3xk array, where k is (number of state transitions) + 1
        heart_rates (list): list of inferred heart rates for each recording
    """
    extracted_segmentations = []
    heart_rates = []
    
    for i in range(len(recordings)):
        if verbose >= 3 and (i+1)%50==0:
            print(f'Processing recording {i+1}/{len(recordings)}...')
        # get lower and upper heartrates
        hr_lims = team_data_processing.define_hr_thresh(ages[i], scale_thresh=1)
        assigned_states, hr = run_segmentation.run_hmm_segmentation(recordings[i], 
                                                                hmm_model['models'], 
                                                                hmm_model['pi_vector'], 
                                                                hmm_model['total_obs_distribution'],
                                                                min_heart_rate=hr_lims[0], 
                                                                max_heart_rate=hr_lims[1], 
                                                                return_heart_rate=True)

        segmentation_indices = team_data_processing.get_segmentation_indices(assigned_states)

        extracted_segmentations.append(segmentation_indices)
        heart_rates.append(hr)
    return extracted_segmentations, heart_rates


def extract_recording_features(recordings, segmentations):
    """
    Parameters
        recordings (list): list of individual recordings
        segmentations (list): lists of segmentations annotation arrays

    Returns
        cc_features (arr): flattened arr of ave_seg_features (4x10), ave_seg_length (4), std_seg_length (4)
    """
    cc_features = []
    #bandsnp.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) 
    bands=np.array([0, 5, 10, 24, 32, 36, 40, 48, 63, 88, 100]) # used as start:end

    for rec, seg in zip(recordings, segmentations):
        # use given segmentations if available
        cc_fts = team_data_processing.get_cardiac_cycle_features(rec, seg, bands)
        cc_features.append(cc_fts)
    
    cc_features = np.vstack(cc_features).astype(np.float32)

    return cc_features