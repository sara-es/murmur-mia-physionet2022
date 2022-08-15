import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

def train_model(cc_features, pt_features, labels, verbose=1):
    X = np.hstack((cc_features, pt_features)).astype(np.float32) # concatenate along axis 1
    y = labels.reshape(-1) # reshape column to row vector

    # using random forest, so normalisation not strictly necessary
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = GradientBoostingClassifier(subsample=0.5, 
                                       max_features=0.5, 
                                       max_depth=5, 
                                      )
    
    if verbose >= 3:
        print(f"Gradboost train 4-cv score: {cross_val_score(model, X, y, cv=4)}")
    model.fit(X, y)
    
    y_pred = model.predict(X)
    if verbose >= 3:
        print(f"Train accuracy score: {accuracy_score(labels,y_pred):.4f}")
    return scaler, model

def run_model(scaler, model, cc_features, pt_features):
    X = np.hstack((cc_features, pt_features))
    X = scaler.transform(X)
    y_pred = model.predict(X)
    probs = model.predict_proba(X)

    return y_pred, probs