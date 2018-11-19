#coding=utf-8
import librosa
import numpy as np
import pandas as pd
import argparse
from pysndfx import AudioEffectsChain
import math
import python_speech_features
import scipy as sp
from scipy import signal
import os
from glob import glob

def invlogamplitude(S):
    return 10.0 ** (S / 10.0)

def reduce_noise_mfcc_down(y, sr):
    hop_length = 512
    n_mfcc = 20

    ## librosa
    mfcc = extract_features_from_song(y, sr, hop_length, n_mfcc)
    mfcc = librosa.mel_to_hz(mfcc)

    ## mfcc
    # mfcc = python_speech_features.base.mfcc(y)
    # mfcc = python_speech_features.base.logfbank(y)
    # mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n ** 2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().highshelf(frequency=min_hz * (-1) * 1.2, gain=-12.0, slope=0.6).limiter(gain=8.0)
    y_speech_boosted = speech_booster(y)

    return (y_speech_boosted)

def trim_silence(y):
    y_trimmed, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)
    return y_trimmed, trimmed_length


def reduce_noise_from_audio(filename):
    x, sr = load_audio_file(filename)
    x_boosted = reduce_noise_mfcc_down(x, sr)
    x_boosted, time_trimmed = trim_silence(x_boosted)
    librosa.output.write_wav(filename, x_boosted, sr)
    

def generate_audio_from_mfcc(input_shape, mfcc, sr, filename, n_mel=128, n_fft=2048):
    # Build reconstruction mappings,
    n_mfcc = mfcc.shape[0]
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    mel_basis = librosa.filters.mel(sr, n_fft)

    # Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))

    # Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfcc)))

    # Impose reconstructed magnitude on white noise STFT.
    excitation = np.random.randn(input_shape)
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))

    # store audio in file
    librosa.output.write_wav(filename, recon, sr)

    # reduce noise in output file
    reduce_noise_from_audio(filename)


# get Mel-frequency cepstral coefficients
def extract_features_from_song(x, sr, hop_length=512, n_mfcc=20):
    # X = librosa.stft(x)
    mfcc = librosa.feature.mfcc(x, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
    return mfcc

# load music file
def load_audio_file(filename):
    try:
        x, sr = librosa.load(filename, res_type='kaiser_fast', offset=0, duration=30)
    except Exception as e:
        print("Error encountered while parsing file: ", filename)
        print(e)
        return None
    
    return x, sr

def generate_training_data(songs):
    data_X = np.empty([0, 20])

    for song in songs:
        x, sr = load_audio_file(song)
        mfcc = extract_features_from_song(x, sr, hop_length=512)
        mfcc = mfcc.transpose()
        data_X = np.concatenate((data_X, mfcc))
    
    X = pd.DataFrame(data_X)
    Y = [X.shift(-1)]
    Y.append(X)
    X = pd.concat(Y, axis=1)
    X.fillna(0, inplace=True)
    data_X = X.values
    x, y = data_X[:, :20], data_X[:, 20:]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    y = y.reshape(y.shape[0], 1, y.shape[1])

    np.save("../results/train_X_3.npy", x)
    np.save("../results/train_Y_3.npy", y)

def main():
    parser = argparse.ArgumentParser(
        prog = 'Music Generation',
        usage = 'To generate new music sequences using MFCC features',
        description = 'python3 mfcc_to_wav.py -i [Input Filename] -o [Output Filename]',
        epilog = 'MIT Licensce',
        add_help=True       
    )
    parser.add_argument('-i', '--input', help = 'Input Filename', required = True)
    # parser.add_argument('-o', '--output', help = 'Output Filename (wav)', required = True)
    args = parser.parse_args()

    input_song_list = [y for x in os.walk(args.input) for y in glob(os.path.join(x[0], '*.mp3'))]

    generate_training_data(input_song_list)

    # create_lstm_network()

    # print(type(mfcc[0]))
    # (#music files, #mfcc, #dimension of each mfcc vector)
    # print(mfcc.shape)

    # generate_training_data(mfcc)

    # use mfcc in ML model

    # generate audio back from mfcc
    # generate_audio_from_mfcc(x.shape[0], mfcc, sr, args.output)

if __name__ == '__main__':
    main()