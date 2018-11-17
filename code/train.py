
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
from keras.optimizers import SGD
np.random.seed(1337)  # for reproducibility

# from keras.layers.core import TimeDistributedDense
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
	batch_size = 25
	hidden_units = 10
	
	X = np.load("../results/train_X.npy")
	y = np.load("../results/train_Y.npy")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
	
	model = Sequential()
	# batch_input_shape= (batch_size, X_train.shape[1], X_train.shape[2])

	# note that it is necessary to pass in 3d batch_input_shape if stateful=True
	# model.add(LSTM(64, return_sequences=True, stateful=False,
				# batch_input_shape= (batch_size, X_train.shape[1], X_train.shape[2])))
	# model.add(LSTM(64, return_sequences=True, stateful=False))
	# model.add(LSTM(64, stateful=False))
	# model.add(Dropout(.25))
	# model.add(Dense(nb_classes, activation='softmax'))

	model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(64))
	model.add(Dropout(0.2))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5, validation_data=(X_test, y_test))

	# for i in range(1000):
	# 	x = numpy.reshape(pattern, (1, len(pattern), 1))
	# 	x = x / float(n_vocab)
	# 	prediction = model.predict(x, verbose=0)
	# 	index = numpy.argmax(prediction)
	# 	result = int_to_char[index]
	# 	seq_in = [int_to_char[value] for value in pattern]
	# 	sys.stdout.write(result)
	# 	pattern.append(index)
	# 	pattern = pattern[1:len(pattern)]
	# y_pred=model.predict_classes(X_test, batch_size=batch_size)


if __name__ == '__main__':
    main()