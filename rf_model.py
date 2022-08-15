import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_rf_model(cc_features, pt_features, labels):
    X = np.hstack((cc_features, pt_features)) # concatenate along axis 1
    y = labels.reshape(-1) # reshape column to row vector

    # using random forest, so normalisation not strictly necessary
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X, y)

    return scaler, model

def run_rf_model(scaler, model, cc_features, pt_features):
    X = np.hstack((cc_features, pt_features))
    X = scaler.transform(X)
    y_pred = model.predict(X)

    return y_pred