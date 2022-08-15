import numpy as np
import team_constants

def probs_to_labels(y_pr):
    # magic number
    return (y_pr > 0.2).astype(int)


def probs_to_one_hot(y_pr):
    # classes = ['Present', 'Unknown', 'Absent']
    thresh_high = team_constants.NN_THRESH_HIGH
    thresh_low = team_constants.NN_THRESH_LOW

    # probs is a single-column array
    present = (y_pr >= thresh_high).astype(int)
    unknown = np.logical_and(y_pr < thresh_high, y_pr >= thresh_low).astype(int)
    absent = (y_pr < thresh_low).astype(int)

    labels = np.hstack([present, unknown, absent])

    return labels


def labels_to_binary(class_labels):
    # labels is array of all labels, with 'Murmur', 'Unknown', 'Absent' as {0,1,2}
    # cast Murmur (0) and Unknown (1) to 1 and Absent (2) to 0
    binary_labels_inv = class_labels//2
    binary_labels = ~binary_labels_inv.astype(bool)
    return binary_labels


def class_labels_to_one_hot(class_label):
    labels = np.zeros((class_label.shape[0],3))
    for i, cl in enumerate(class_label.astype(int)):
        labels[i, cl] = 1
    return labels


def calc_accuracy(y_pred, y_true):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    accuracy = np.mean(y_true==y_pred)
    return accuracy


def calc_sensitivity(y_pred, y_true):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sens = np.sum(np.logical_and(y_true==1, y_pred==1))/np.sum(y_true)
    return sens


def calc_confusion_matrix(y_pred, y_true):
    conf_matrix = np.zeros((2, 2))
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    # TP
    conf_matrix[0,0] = np.sum(np.logical_and(y_true==1, y_pred==1))
    # FP
    conf_matrix[0,1] = np.sum(np.logical_and(y_true==0, y_pred==1))
    # FN
    conf_matrix[1,0] = np.sum(np.logical_and(y_true==1, y_pred==0))
    # TN
    conf_matrix[1,1] = np.sum(np.logical_and(y_true==0, y_pred==0))
    return conf_matrix

def get_ngm(array):
    # normalized geometric mean
    with np.errstate(all='ignore'):
        gm = np.exp(np.mean(np.log(array), axis=0))
    ngm = gm / gm.sum()
    return ngm