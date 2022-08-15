import numpy as np

# recording_to_murmur_predictions thresholds
OUTLIER_THRESHOLD = 0.9
PRESENT_THRESHOLD = 0.4
ABSENT_THRESHOLD = 0.6
POS_VETO_THRESHOLD = 0.62

# recording_to_outcome_predictions thresholds
MOD_OUTCOME_THRESHOLD = .68
OUTL_OUTCOME_THRESHOLD = .7

# min and max values for imputation
MIN_AGE = 0.5 # in months
MAX_AGE = 250.
MIN_HEIGHT = 30. # in cm. 35 in test data
MAX_HEIGHT = 190. # 180 in test data 
MIN_WEIGHT = 2.0 # in kg. 2.3 in test
MAX_WEIGHT = 120. # 110.8 in test

# [age, female, male, height, weight, is_pregnant]
MINS = [MIN_AGE, 0.0, 0.0, MIN_HEIGHT, MIN_WEIGHT, 0.0]
MAXS = [MAX_AGE, 1.0, 1.0, MAX_HEIGHT, MAX_WEIGHT, 1.0]

# time to remove from end of recordings in seconds*4kHz
CUTOFF_TIME = 1 * 4000

# if setting all recordings to standard length, in seconds*4kHz
STD_REC_LENGTH = 20 * 4000