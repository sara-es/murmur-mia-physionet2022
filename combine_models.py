import numpy as np
import team_constants, metrics
from sklearn.linear_model import LogisticRegression


def recording_to_murmur_predictions(outlier_probs_per_rec, model_probs_per_rec,
                                    unknown_threshold=team_constants.OUTLIER_THRESHOLD,
                                    pres_threshold=team_constants.PRESENT_THRESHOLD,
                                    abs_threshold=team_constants.ABSENT_THRESHOLD,
                                    pos_veto_threshold=team_constants.POS_VETO_THRESHOLD,
                                    # score_threshold=team_constants.MURMUR_WEIGHTS
                                    ):
    """
    Parameters
        outlier_probs_per_recording: list of shape(n_recordings, 3)
        cnn_probs_per_rec: list of shape(n_recordings, 3)
        gb_probs_per_rec: list of shape(n_recordings, 3)
        thresholds: set in team_constants

    Returns
        murmur_pred: len(3) one-hot encoded array
        murmur_probs: len(3) array of [Murmur, Unknown, Absent] probabilities
    
    """
    murmur_pred = np.zeros(3)
    murmur_probs = metrics.get_ngm(model_probs_per_rec)

    if outlier_probs_per_rec.ndim <= 1: # make dims line up if there's only one recording for that patient
        outlier_probs_per_rec = outlier_probs_per_rec.reshape(1,-1)
    outlier_probs = np.mean(outlier_probs_per_rec, axis=0)

    if (model_probs_per_rec[:,0] > pos_veto_threshold).any(): #if any of the recordings are very confident
        murmur_pred[0] = 1 # assign present
    elif outlier_probs[1] > unknown_threshold: # let the unknown model have a go first
        murmur_pred[1] = 1 # assign unknown
    elif murmur_probs[2] > abs_threshold: # if confidence of Absent is above the threshold
        murmur_pred[2] = 1 # assign Absent
    elif murmur_probs[0] > pres_threshold:
        murmur_pred[0] = 1 # assign present
    else:
        murmur_pred[1] = 1 # unknown

    return murmur_pred, murmur_probs


def recording_to_outcome_predictions(outlier_probs_per_recording,
                                    model_probs_per_recording,
                                    outl_threshold=team_constants.OUTL_OUTCOME_THRESHOLD,
                                    model_threshold=team_constants.MOD_OUTCOME_THRESHOLD):
    """
    Parameters
        outlier_probs_per_recording: list of shape(n_recordings, 2)
        gb_probs_per_recording: list of shape(n_recordings, 2)
        threshold: threshold below which assign Abnormal, above which Normal

    Returns
        outcome_pred: len(2) one-hot encoded array
        outcome_probs: len(2) array of [Abnormal, Normal] probabilities
    
    """
    # one hot encoding from output labels
    outcome_pred = np.zeros(2)
    avg_outlier_prob = metrics.get_ngm(outlier_probs_per_recording)
    avg_model_prob = metrics.get_ngm(model_probs_per_recording)
        
    if avg_model_prob[1] > model_threshold: # if confidence of Normal is above threshold 0.525 in best 
        pred_cc = False # assign normal
    else: 
        pred_cc = True
    if avg_outlier_prob[1] > outl_threshold: # if confidence of Normal is above threshold
        pred_feat = False # assign normal
    else: 
        pred_feat = True    
        
    if pred_cc or pred_feat: # if either is Abnormal
        outcome_pred = np.array([1, 0]) # assign Abnormal
    else:
        outcome_pred = np.array([0, 1])

    # just take the average, why not
    outcome_probs = np.mean(np.vstack((avg_outlier_prob, avg_model_prob)), axis=0) 

    return outcome_pred, outcome_probs


def combine_CNN_GB(CNN_probs, GB_probs):
    X = np.hstack((CNN_probs, GB_probs))
    
    # set up model
    model_intercept = np.array([1.32173036, 0.04718764, -1.368918], dtype = 'float32')
    model_classes = np.array([0., 1., 2.], dtype = 'float32')
    model_coeffs = np.array([[ 2.02216242, 0., -2.02173499, 2.15213884, -1.95735788, -0.19435353],
     [ 0.6068475, 0., -0.60663192, -1.40878808, 1.74781421, -0.33881054],
     [-2.62900992, 0., 2.62836691, -0.74335076, 0.20954368, 0.53316407]])
    
    model = LogisticRegression()
    model.coef_ = model_coeffs
    model.intercept_ = model_intercept
    model.classes_ = model_classes
    
    label = model.predict(X)
    probs = model.predict_proba(X)
    
    return label, probs

# def combine_outcome_models(outlier_pred, outcome_pred_gb, outcome_probs_gb):
#     # if the class is unknown murmur, set the outcome to Abnormal, otherwise leave it alone
#     if outlier_pred[1] == 1:
#         outcome_pred = [1,0]
#         outcome_probs = [1,0]
#     else:
#         outcome_pred = outcome_pred_gb
#         outcome_probs = outcome_probs_gb
#     return outcome_pred, outcome_probs
