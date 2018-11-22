import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, IPython.display
import librosa.display
import os, glob
import h5py
from IPython.lib.display import Audio

def invlogamplitude(S):
    return 10.0**(S/10.0)


data = numpy.load("mfcc_result5.npy")
sr=22050

mfccs = data[0]
min_x = -678.8547296660762
max_x = 264.0727396648673

max_1 = 200
min_1 = -400
#mfccs = sklearn.preprocessing.scale(mfccs, axis=1)


filename = 'classical/classical.00001.wav'
y, sr = librosa.load(filename)



mfccs_actualsong = librosa.feature.mfcc(y, sr=sr)

sc = sklearn.preprocessing.MinMaxScaler(feature_range=(-2,2))
sc = sc.fit(mfccs_actualsong)
mfccs = sc.inverse_transform(mfccs)


def generate_audio_from_mfcc(y, mfcc, sr, filename, n_mel=128, n_fft=2048):
    # Build reconstruction mappings,
    n_mfcc = mfcc.shape[0]
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    mel_basis = librosa.filters.mel(sr, n_fft)

    # Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/numpy.maximum(0.0005, numpy.sum(numpy.dot(mel_basis.T, mel_basis),axis=0))

    # Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, numpy.newaxis] * numpy.dot(mel_basis.T, invlogamplitude(numpy.dot(dctm.T, mfcc)))

    # Impose reconstructed magnitude on white noise STFT.
    excitation = numpy.random.randn(y.shape[0])
    E = librosa.stft(excitation)
    recon = librosa.istft(E/numpy.abs(E)*numpy.sqrt(recon_stft))

    # store audio in file
    librosa.output.write_wav(filename, recon, sr)

    # reduce noise in output file
    # reduce_noise_from_audio(filename)
    return
    

generate_audio_from_mfcc(y,mfccs,sr,'output26_fromresults5_tp1.wav')

librosa.display.specshow(mfccs, sr=sr, x_axis='time')
librosa.display.specshow(mfccs_actualsong, sr=sr, x_axis='time')

'''
mfccs_actualsong = sklearn.preprocessing.scale(mfccs_actualsong, axis=1)
librosa.display.specshow(mfccs_actualsong, sr=sr, x_axis='time')
'''

'''
sc = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler = sc.fit(mfccs_actualsong)
data_scaled = scaler.transform(mfccs_actualsong)
librosa.display.specshow(data_scaled, sr=sr, x_axis='time')
data_back = sc.inverse_transform(data_scaled)
librosa.display.specshow(data_back, sr=sr, x_axis='time')
'''