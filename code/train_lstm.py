import numpy as np
np.random.seed(1337)

from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    X = np.load("../results/train_X_sample.npy")
    y = np.load("../results/train_Y_sample.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_epoch = 5
    n_batch = 64
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True, stateful=False, recurrent_dropout=0.02))
    model.add(LSTM(128, return_sequences=True, stateful=False, recurrent_dropout=0.02))
    model.add(LSTM(64, return_sequences=True, stateful=False, recurrent_dropout=0.02))
    model.add(TimeDistributed(Dense(20, activation='relu')))
    model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch, validation_data=(X_test, y_test), verbose=2, shuffle=False)

    with open('../results/lstm_model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('../results/lstm_model_weights.h5')

    loss, accuracy = model.evaluate(X_test, y_test)
    print(loss, accuracy)

if __name__ == '__main__':
    main()