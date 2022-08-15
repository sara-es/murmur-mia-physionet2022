import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import cross_val_score
from features import get_all_features

def train_model_outcomes(recordings, murmur_labels):
    # labels are murmur_labels, {0, 1, 2}
    labels = murmur_labels.reshape(-1) # reshape column to row vector
    features = []
    
    for rec in recordings:
        features.append(get_all_features(rec))
    features = np.vstack(features)
    
    model = GradientBoostingClassifier()  
    model.fit(features, labels)
    
    return model


def run_model_outcomes(model, recordings): 
    features = []
    for rec in recordings:
        features.append(get_all_features(rec))
    features = np.vstack(features)
    
    # returns an array of pred/prob for each recording
    preds = model.predict(features)
    probs = model.predict_proba(features)
    return preds, probs


def train_model(recordings, murmur_labels, num_recs, heart_rates):
    # labels are murmur_labels, {0, 1, 2}
    labels = murmur_labels.reshape(-1) # reshape column to row vector
    heart_rates = heart_rates.reshape(-1,1) # reshape row to column vector ¯\_(ツ)_/¯
    features = []
    
    for rec in recordings:
        features.append(get_all_features(rec))
    features = np.vstack(features)
    features = np.concatenate((features, num_recs, heart_rates), axis = 1)
    
    model = GradientBoostingClassifier()  
    model.fit(features, labels)
    
    return model


def run_model(model, recordings, num_recs, heart_rates): 
    heart_rates = heart_rates.reshape(-1,1) # reshape row to column vector
    features = []
    for rec in recordings:
        features.append(get_all_features(rec))
    features = np.vstack(features)
    features = np.concatenate((features, num_recs, heart_rates), axis = 1)
    
    # returns an array of pred/prob for each recording
    preds = model.predict(features)
    probs = model.predict_proba(features)
    return preds, probs