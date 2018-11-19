import argparse
import numpy as np
np.random.seed(1337)

from mfcc_to_wav import generate_audio_from_mfcc, reduce_noise_from_audio
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

def generate(X_test):
    # load model
    f = open('../results/lstm_model.json', 'r')
    model = model_from_json(f.read())
    f.close()
    model.load_weights('../results/lstm_model_weights.h5')
    model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])

    # generates 30s worth of new sequences
    seq_length = 1292
    start = np.random.randint(0, len(X_test)-1)
    pattern = np.reshape(X_test[start], (1, 1, 20))
    pred_opt = np.empty([0, 20])
    
    for i in range(seq_length):
        pred = model.predict(pattern, batch_size=1, verbose=0)
        pattern = pred
        pred = np.reshape(pred, (1, 20))
        pred_opt = np.concatenate((pred_opt, pred))
    
    print("Generated new sequences!")
    return pred_opt

def main():
    parser = argparse.ArgumentParser(
        prog = 'Music Generation',
        usage = 'To generate new music sequences using MFCC features',
        description = 'python3 generate.py -o [Output Filename]',
        epilog = 'MIT Licensce',
        add_help=True       
    )
    parser.add_argument('-o', '--output', help = 'Output Filename (wav)', required = True)
    args = parser.parse_args()

    # load data
    X = np.load("../results/train_X_sample.npy")
    y = np.load("../results/train_Y_sample.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # generate new music sequences
    new_seq_mfcc = generate(X_test).transpose()

    # generate audio back from mfcc
    song_length = 30
    sampling_rate = 22050
    song_freq_size = song_length * sampling_rate
    generate_audio_from_mfcc(song_freq_size, new_seq_mfcc, sampling_rate, args.output)
    reduce_noise_from_audio(args.output)
    print("Generation Complete!")


if __name__ == '__main__':
    main()
