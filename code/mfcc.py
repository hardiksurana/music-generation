import librosa
import numpy as np
from pipes import quote
import os

# # displays graph of *.wav, *.au files
# def display_mfcc(song):
#     y, _ = librosa.load(song)
#     mfcc = librosa.feature.mfcc(y)

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
#     plt.colorbar()
#     plt.title(song)
#     plt.tight_layout()
#     plt.show()

def convert_mp3_to_wav(filename, sample_frequency):
	ext = filename[-4:]
	if(ext != '.mp3'):
		return
	files = filename.split('/')
	orig_filename = files[-1][0:-4]
	orig_path = filename[0:-len(files[-1])]
	new_path = ''
	if(filename[0] == '/'):
		new_path = '/'
	for i in xrange(len(files)-1):
		new_path += files[i]+'/'
	tmp_path = new_path + 'tmp'
	new_path += 'wave'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	if not os.path.exists(tmp_path):
		os.makedirs(tmp_path)
	filename_tmp = tmp_path + '/' + orig_filename + '.mp3'
	new_name = new_path + '/' + orig_filename + '.wav'
	sample_freq_str = "{0:.1f}".format(float(sample_frequency)/1000.0)
	cmd = 'lame -a -m m {0} {1}'.format(quote(filename), quote(filename_tmp))
	os.system(cmd)
	cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(filename_tmp), quote(new_name), sample_freq_str)
	os.system(cmd)
	return new_name


def extract_features_song(f):
    try:
        X, sampling_rate = librosa.load(f, res_type='kaiser_fast')

        # get Mel-frequency cepstral coefficients
        mfcc = librosa.feature.mfcc(X, sr=sampling_rate, n_mfcc=40)
        # normalize values between -1,1 (divide by max)
        mfcc /= np.amax(np.absolute(mfcc))
    except Exception as e:
        print("Error encountered while parsing file: ", f)
        return None
    return np.ndarray.flatten(mfcc)

res = extract_features_song('../datasets/songs/twenty one pilots_ Ride (Video).mp3')
print(res)
print(res.shape)
