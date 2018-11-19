
# def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
# 	model = Sequential()
# 	#This layer converts frequency space to hidden space
# 	model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
# 	for cur_unit in xrange(num_recurrent_units):
# 		model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
# 	#This layer converts hidden space back to frequency space
# 	model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
# 	model.compile(loss='mean_squared_error', optimizer='rmsprop')
# 	return model

import numpy as np
np.random.seed(1337)

from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
	X = np.load("../results/train_X_3.npy")
	y = np.load("../results/train_Y_3.npy")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	return X_train, X_test, y_train, y_test

def lstm_model(X_train, X_test, y_train, y_test):
	n_batch = 256
	n_epoch = 5
	
	model = Sequential()
	model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, stateful=False, recurrent_dropout=0.2))
	model.add(LSTM(64, return_sequences=True, stateful=False, recurrent_dropout=0.2))
	model.add(LSTM(64, return_sequences=True, stateful=False, recurrent_dropout=0.2))
	model.add(TimeDistributed(Dense(20, activation='relu')))
	model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])
	print(model.summary())	
	model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch, validation_data=(X_test, y_test), verbose=2, shuffle=False)

	return model

def main():	
	X_train, X_test, y_train, y_test = data()
	lstm_model(X_train, X_test, y_train, y_test)

	yhat = model.predict(X_test, batch_size=1)
	y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[2]))
	yhat = np.reshape(yhat, (yhat.shape[0], yhat.shape[2]))

	print(mean_squared_error(y_test, yhat))

	# for i in range(len(X_test)):
	# 	testX, testy = X_test[i], y_test[i]
	# 	testX = testX.reshape(1, 1, 20)
	# 	yhat = model.predict(testX, batch_size=1)
	# 	print(mean_squared_error(testy, yhat[0]))
	# 	break


if __name__ == '__main__':
    main()