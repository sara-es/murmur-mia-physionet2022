# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 20:26:55 2022

@author: g51388dw
"""

#combine gradboost and ensemble models

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

cnn_file = 'C:\Downloads\\ensembled_cnns_with_nn_20220812-1019.sav'
gb_file = 'C:\Downloads\\results_gradboost_cv20220808_1830.sav'
outlier_file = 'C:\Downloads\\results_outlier_on_outcomesdemographics_cv_20220809_1737.sav'

cnn = pickle.load(open(cnn_file, 'rb'))
gb = pickle.load(open(gb_file, 'rb'))
outlier = pickle.load(open(outlier_file, 'rb'))

# train model to balance between cnn_ensemble and gradboost models

# get all the labels
target_dict = ['split 0 targets', 'split 1 targets', 'split 2 targets', 'split 3 targets', 'split 4 targets']
all_labels = []
for targ in target_dict:
    labs = cnn[targ]
    labs = np.hstack(labs)
    all_labels.append(labs)
all_labels = np.hstack(all_labels)
    
cnn_dict = ['split 0 cnn probabilities', 'split 1 cnn probabilities', 'split 2 cnn probabilities', 'split 3 cnn probabilities', 'split 4 cnn probabilities']
all_cnn_probs = []
for targ in cnn_dict:
    cnn_probs = np.vstack(cnn[targ])
    all_cnn_probs.append(cnn_probs) 
all_cnn_probs = np.vstack(all_cnn_probs)

gb_dict = ['split 0 gb murmur probabilities', 'split 1 gb murmur probabilities', 'split 2 gb murmur probabilities', 'split 3 gb murmur probabilities', 'split 4 gb murmur probabilities']
all_gb_probs = []
for targ in gb_dict:
    gb_probs = np.vstack(gb[targ])
    all_gb_probs.append(gb_probs) 
all_gb_probs = np.vstack(all_gb_probs)


y = all_labels
X = np.hstack((all_cnn_probs, all_gb_probs))

model = LogisticRegression(class_weight={ 0:5, 1:3, 2:1 })
model.fit(X, y)
print(model.coef_)
print(model.intercept_)


# kf = KFold(n_splits=5, shuffle=True)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
    
#     model = LogisticRegression()
#     model.fit(X_train, y_train)
#     print(model.coef_)
#     preds = model.predict(X_test)
#     print(accuracy_score(y_test,preds))

#     cnn_pred = np.argmax(all_cnn_probs[test_index], axis = 1)
#     print(accuracy_score(y_test,cnn_pred))
    
#     gb_pred = np.argmax(all_gb_probs[test_index], axis = 1)
#     print(accuracy_score(y_test,gb_pred))


# print(model.coef_)



# kf = KFold(n_splits=5, shuffle=True)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
    
#     model = LogisticRegression()
#     model.fit(X_train, y_train)
#     print(model.coef_)
    
#     preds = model.predict(X_test)
#     print(accuracy_score(y_test,preds))

#     cnn_pred = np.argmax(all_cnn_probs[test_index], axis = 1)
#     print(accuracy_score(y_test,cnn_pred))
    
#     gb_pred = np.argmax(all_gb_probs[test_index], axis = 1)
#     print(accuracy_score(y_test,gb_pred))
