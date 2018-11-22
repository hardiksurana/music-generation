# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 19:52:14 2018

@author: Akhil
"""

import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, IPython.display
import librosa.display
import os, glob
import h5py
import pandas


filepath_music = 'classical/'
signals = []
output_numpy_name = 'mfcc_array'




for filename in glob.glob(os.path.join(filepath_music, '*.wav')):
    signals.append(librosa.load(filename)[0])
    

def extract_mfcc_from_music(signal):
    return librosa.feature.mfcc(signal)
    

#x = extract_mfcc_from_music(signals[0])
#print(x)
mfcc_list = []
for x in signals:
   tp = extract_mfcc_from_music(x)
   print(tp.shape)
   if tp.shape != (20,1293):
       continue
   #tp = sklearn.preprocessing.scale(tp, axis=1)
   sc = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
   sc = sc.fit(tp)
   tp = sc.transform(tp)
   mfcc_list.append(tp)
    
   
#print(len(mfcc_list[0][0]))
mfcc_array = numpy.array(mfcc_list)
#for x in signals:
#    print(extract_mfcc_from_music(x))

print(mfcc_array.shape)

numpy.save('mfcc_array5x1n'+'.npy',mfcc_array)

 