import librosa # for working with audio in python
import librosa.display # for waveplots, spectograms, etc
import soundfile as sf # for accessing file information
import IPython.display as ipd # for playing files within python
import numpy as np
import pandas as pd

def convert(audio_file):
    # sr is sampling rate - currently default
    # The sampling rate must be twice the frequency of the highest frequency
    # that is desired to be captured according to the Nyquist-Shannon sampling theorem.
    x, sample_rate = librosa.load(audio_file, sr=None)

    mfccs = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40)
    return mfccs
    
