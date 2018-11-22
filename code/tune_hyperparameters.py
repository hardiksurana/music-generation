import numpy as np
np.random.seed(1337)

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def data():
	X = np.load("../results/train_X_sample_2_classes.npy")
	y = np.load("../results/train_Y_sample_2_classes.npy")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	return X_train, X_test, y_train, y_test

def lstm_model(X_train, X_test, y_train, y_test):
	# n_epoch = 100
	input_shape = (X_train.shape[1], X_train.shape[2])

	recurrent_dropout = {{uniform(0, 1)}}
	n_cells = {{choice([64, 128])}}
	n_epoch = {{choice([50, 100, 200])}}
	optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}}
	activation = {{choice(['relu', 'tanh'])}}
	n_batch = {{choice([32, 64, 128])}}

	params = {
        'recurrent_dropout': recurrent_dropout,
        'n_cells': n_cells,
		'optimizer': optimizer,
		'activation': activation,
		'n_batch': n_batch,
		'n_epoch': n_epoch
    }

	model = Sequential()
	model.add(LSTM(n_cells, input_shape=input_shape, return_sequences=True, stateful=False, recurrent_dropout=recurrent_dropout))
	model.add(LSTM(n_cells, return_sequences=True, stateful=False, recurrent_dropout=recurrent_dropout))
	model.add(LSTM(n_cells, return_sequences=True, stateful=False, recurrent_dropout=recurrent_dropout))
	model.add(TimeDistributed(Dense(20, activation=activation)))
	model.compile(loss='mean_squared_error',optimizer=optimizer, metrics=['accuracy'])
	print(model.summary())
	result = model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch, validation_data=(X_test, y_test), verbose=2, shuffle=False)
	validation_acc = np.amax(result.history['val_acc']) 
	print('Best validation accuracy: ', validation_acc)
	return {
		'loss': -validation_acc, 
		'status': STATUS_OK, 
		'model': model,
		'model_params': params,
		}


def main():	
	best_run, best_model = optim.minimize(model=lstm_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

	X_train, X_test, y_train, y_test = data()
	print(best_run)

if __name__ == '__main__':
    main()