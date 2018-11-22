import numpy as np
np.random.seed(1337)

from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model


def main():
    X = np.load("../results/train_X_classical.npy")
    y = np.load("../results/train_Y_classical.npy")
    n_songs = 100

    '''
    X_train = np.empty([0, 1, 20])
    y_train = np.empty([0, 1, 20])
    X_test = np.empty([0, 1, 20])
    y_test = np.empty([0, 1, 20])

    train_size_per_song = int((X.shape[0] / n_songs) * 0.8)

    for i in range(n_songs):
        index_start = i * 1292
        index_end = (i + 1) * 1292
        X_train = np.concatenate((X_train, X[index_start : (index_start + train_size_per_song), :, :]), axis=0)
        y_train = np.concatenate((y_train, y[index_start : (index_start + train_size_per_song), :, :]), axis=0)

        X_test = np.concatenate((X_test, X[(index_start + train_size_per_song) : index_end, :, :]), axis=0)
        y_test = np.concatenate((y_test, y[(index_start + train_size_per_song) : index_end, :, :]), axis=0)

    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # X_train = X[: int(X.shape[0] * 0.8), :, :]
    # X_test = X[int(X.shape[0] * 0.8) : , :, :]
    # y_train = y[: int(y.shape[0] * 0.8), :, :]
    # y_test = y[int(y.shape[0] * 0.8) : , :, :]
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    n_epoch = 50
    n_batch = 64
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True, stateful=False, activation='linear', recurrent_dropout=0.5))
    model.add(LSTM(128, return_sequences=True, stateful=False, activation='linear', recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(20, activation='linear')))
    sgd = optimizers.SGD(lr=0.1, decay=1e-2, momentum=0.9, nesterov=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=2)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=5, patience=5, verbose=2)
    model.compile(loss='mean_squared_error',optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='../results/model_architecture.png', show_shapes=True, show_layer_names=True)
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch, validation_data=(X_test, y_test), verbose=2, shuffle=False, callbacks=[early_stop])

    with open('../results/lstm_model_sgd_3.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('../results/lstm_model_sgd_weights_3.h5')

    loss, accuracy = model.evaluate(X_test, y_test)
    print(loss, accuracy)
    

if __name__ == '__main__':
    main()