# Python classifier code from team Murmur Mia! for the George B. Moody PhysioNet Challenge 2022

**This is the submission selected for evaluation on the hidden test set. Overall this code achieved a weighted murmur score of 0.755, ranking 9/39, and an outcome score of 14228, ranking 28/39, on the test data.**

This code is available for reuse and modification under the [2-Clause BSD License](https://opensource.org/licenses/BSD-2-Clause). If you find any portion of it helpful, we ask that you please cite our accompanying paper: [Two-stage Classification for Detecting Murmurs from Phonocardiograms Using Deep and Expert Features]().

> Summerton, S., Wood, D., Murphy, D., Redfern, O., Benatan, M., Kaisti, M., & Wong, D. C. (2022). Two-stage Classification for Detecting Murmurs from Phonocardiograms Using Deep and Expert Features. In 2022 Computing in Cardiology (CinC), volume 49. IEEE, 2023; 1–4.

## Model and hyperparameter description

09. Super-ensemble of CNNs + gradient boosting + gboost for murmur outliers, using *weighted* logistic regression to set thresholds; 2-stage gradient boosting for outcomes (with num recordings, heart rates features)

**Models used:**
  * outlier_detector_murmur: gradient boosing on recording features, number of recordings, and heart rates 
  * rubin_cnn_ensemble: 5 CNNs trained on murmur targets, ensembled using a NN
  * linear regression to combine above, with positive weights on Present (5) and Unknown (3) labels
  * outlier_detector_outcome: gradient boosting on recording features, number of recordings, and heart rates
  * gb_outcome: gradient boosting on outcomes

**recording_to_murmur_predictions thresholds**
  * OUTLIER_THRESHOLD = 0.9
  * PRESENT_THRESHOLD = 0.4
  * ABSENT_THRESHOLD = 0.6
  * POS_VETO_THRESHOLD = 0.62

**recording_to_outcome_predictions thresholds**
  * MOD_OUTCOME_THRESHOLD = .68
  * OUTL_OUTCOME_THRESHOLD = .7
