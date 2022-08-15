import librosa
import numpy as np


def get_mfcc(patient_recordings, sample_rate=4000, n_fft=1024, win_length=None, hop_length=512, n_mels=256, n_mfcc=256):
    # Note: This converts the raw audio recordings into an mfcc cepstrum (following Rubin's paper)

    # get the MFCC cepstrum
    mel_spec = []
    mfcc = []
    for recording in patient_recordings:
        melspec = librosa.feature.melspectrogram(
            y=recording.astype(float),
            sr=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            htk=True,
            norm="slaney",
        )

        mfcc_librosa = librosa.feature.mfcc(
            S=librosa.core.spectrum.power_to_db(melspec),
            n_mfcc=n_mfcc,
            dct_type=2,
            norm="ortho",
        )

        # standardise mfcc
        min_a = np.amin(mfcc_librosa)
        max_a = np.amax(mfcc_librosa)
        mfcc_librosa = (mfcc_librosa - min_a) / (max_a - min_a)

        # standardise mel spectrogram
        min_a = np.amin(melspec)
        max_a = np.amax(melspec)
        melspec = (melspec - min_a) / (max_a - min_a)

        mel_spec.append(melspec)
        mfcc.append(mfcc_librosa)
    return mel_spec, mfcc


def truncate_spectrograms(mel_spec, length):
    for idx, spec in enumerate(mel_spec):
        if spec.shape[1] < length:
            # pad with zeros
            pad_length = length - spec.shape[1]
            mel_spec[idx] = np.pad(spec, ((0, 0), (0, pad_length)), 'constant')
        else:
            mel_spec[idx] = spec[:, :length]

    return mel_spec


def mono_2_color(mel_spec):
    for idx, spec in enumerate(mel_spec):
        mel_spec[idx] = np.stack((spec,) * 3, axis=-1)
        return mel_spec
