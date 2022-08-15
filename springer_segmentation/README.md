# Hidden Markov Model based segmentation for Phonocardiograms

Python implementation of https://github.com/davidspringer/Springer-Segmentation-Code 

## Minimal Example

I haven't actually run this... so yeah, beware. If it's wrong, the `main()` of `train_segmentation.py` definitely works but is a bit convoluted

```python
from springer_segmentation.utils import get_wavs_and_tsvs
from springer_segmentation.train_segmentation import train_hmm_segmentation
from springer_segmentation.run_segmentation import run_hmm_segmentation

# get list of recordings and corresponding segmentations (in the format given in the tsv)
wavs, tsvs = get_wavs_and_tsvs("tiny_test")

# train the model
models, pi_vector, total_obs_distribution = train_hmm_segmentation(wavs, tsvs)

# get segmentations out of the model for the first wav file in our list
annotation, heart_rate = run_hmm_segmentation(wavs[0],
                                      models,
                                      pi_vector,
                                      total_obs_distribution,
                                      min_heart_rate=60,
                                      max_heart_rate= 200,
                                      return_heart_rate=True)
```