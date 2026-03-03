import librosa
import numpy as np
import torchaudio


def extract_mfcc(file_path, n_mfcc, fixed_time_steps):
    """
    Convert audio file into fixed-length MFCC feature matrix.
    """

    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.numpy().squeeze()

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    mfcc = mfcc.T

    if mfcc.shape[0] < fixed_time_steps:
        return None

    return mfcc[:fixed_time_steps]