# Python classifier code from team Murmur Mia! for the George B. Moody PhysioNet Challenge 2022

This is the ninth submission for this team.

09. Super-ensemble of CNNs + gradient boosting + gboost for murmur outliers, using *weighted* logistic regression to set thresholds; 2-stage gradient boosting for outcomes (with num recordings, heart rates features)

Models used:
outlier_detector_murmur: gradient boosing on recording features, number of recordings, and heart rates 
rubin_cnn_ensemble: 5 CNNs trained on murmur targets, ensembled using a NN
linear regression to combine above, with positive weights on Present (5) and Unknown (3) labels
outlier_detector_outcome: gradient boosting on recording features, number of recordings, and heart rates
gb_outcome: gradient boosting on outcomes

# recording_to_murmur_predictions thresholds
OUTLIER_THRESHOLD = 0.9
PRESENT_THRESHOLD = 0.4
ABSENT_THRESHOLD = 0.6
POS_VETO_THRESHOLD = 0.62

# recording_to_outcome_predictions thresholds
MOD_OUTCOME_THRESHOLD = .68
OUTL_OUTCOME_THRESHOLD = .7