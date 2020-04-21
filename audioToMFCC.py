import librosa # for working with audio in python
import librosa.display # for waveplots, spectograms, etc
import soundfile as sf # for accessing file information
import IPython.display as ipd # for playing files within python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
To set window to 25 milliseconds, multiply SR by 0.025
SR is the number of samples in one second

To get # of samples in 25 ms, scale by 0.025

To get stride of 10 ms, multiple sr by 0.01

mfccs = librosa.feature.mfcc(y, sr, n_mfcc=13, hop_length=int(0.010sr), n_fft=int(0.025sr))
librosa.feature.mfcc(y=y, sr=sr, hop_length=int(sr/100), n_fft=int(sr/40))

In order to use MFCCs the architecture of the network has to be modified. In
the following lines this structure will be described. First, MFCCs are computed per
each signal during the pre-processing.To calculate the MFCCs we used the built in
function from the librosa library with 20 MFCCs, frames of 2048 samples and a
hop size of 512. We selected only 12 MFCCs, from the 2nd to the 13rd as they
are the ones that give the most relevant information. This information is upsampled using nearest neighbour and stored in a .json file as it could be converted
directly to a python dictionary. When all the files have been analysed, the network will start the training. In that case, local_condition_batch, lc_filtweights
and lc_gateweights from figure 3.5 will have always 12 channels. Is important to
mention that here MFCCs are directly used to train the network and to generate
new samples. That means that during the generation a .json file containing the 12
MFCCs per sample have to be passed to the network.


So let's say, sampling at sr=44100 frames per second, we end up with:
14848000 : which corresponds to the length of our music (after dividing by sr and then 60 to convert into seconds)

that means 1 sec, we're gonna sample 44100
and there are 336.68 seconds worth of 44100 samples

We can't operate at any more minute than 1 frame, so basically, assuming 30 fps, no less than .03 seconds

If we do 3 frames, that's .1 seconds

so equivalently, we sample the audio at .1 the rate of the X per second sampling
"""
def convert(audio_file):
    # sr is sampling rate - currently default
    # The sampling rate must be twice the frequency of the highest frequency
    # that is desired to be captured according to the Nyquist-Shannon sampling theorem.
    x, sample_rate = librosa.load(audio_file, sr=None)

    total_samples = x.shape[0]
    samples_per_second = sample_rate
    frames_per_sec = 24

    samples_per_frame = total_samples / (frames_per_sec * total_samples / samples_per_second)

    print(samples_per_frame)

    # sf.write("30fps_timestamp.wav", x[int(samples_per_frame * 153):], sample_rate)
    sf.write("original_timestamp.wav", x[int(44100 * 6.37):], sample_rate)
    '''
    print(samples_per_frame * 153)
    print(44100 * 6.37)
    '''
    '''
    # 14848000 / 29000 = 512 (which is also the hop-length)
    # thus, each of the 29000 frames corresponds to 512

    # what is 512?
    # what exactly is 29000?

    # 14848000 / 44100 = 336 something seconds
    # 14848000 / 512 = 29000 ... seconds?

    
    mfccs = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40) 
    # reconstructed = librosa.feature.inverse.mfcc_to_audio(mfccs)
    # sf.write("reconstructed_jardy_VEE5qqDPVGY.wav", reconstructed, sample_rate)

    sf.write("30fps_timestamp.wav", x[int(samples_per_frame * 153):], sample_rate)
    '''
    '''
    small_mfccs = librosa.feature.mfcc(y=x[:int(sample_rate * .1 * 5)], sr=sample_rate, n_mfcc=40)
    print(small_mfccs.shape)

    plt.figure(figsize=(8,8))

    # 4410 / 512 is approx 9 (rounded up)
    # so each sample of 4410 corresponds to [40, 9]
    # or, 0.1 seconds of audio = [40, 9] if we sample at 4410
    librosa.display.specshow(small_mfcc, sr=sample_rate, x_axis='time')
    plt.savefig('MFCCs.png')
    '''
    # return mfccs
    

convert('../jardy_VEE5qqDPVGY.mp3')
