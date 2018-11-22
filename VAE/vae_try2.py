# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:42:24 2018

@author: Akhil
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import load_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot





original_dim = 25860
intermediate_dim1 = 2580
intermediate_dim2 = 256
latent_dim = 128
batch_size = 6
epochs = 50
epsilon_std = 1.0


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


decoder = Sequential([
    Dense(intermediate_dim2, input_dim=latent_dim, activation='relu'),
    Dense(intermediate_dim1, input_dim=intermediate_dim2, activation='relu'),
    #Dense(intermediate_dim1, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
])

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim1, activation='relu')(x)
h2 = Dense(intermediate_dim2, activation='relu')(h)

z_mu = Dense(latent_dim)(h2)
z_log_var = Dense(latent_dim)(h2)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)


data = np.load('mfcc_array5x2n.npy')
data2 = []

for i in range(len(data)):
    tp  = np.reshape(data[i],25860)
    data2.append(tp)
    
t_data = np.array(data2)

x_train, x_test = train_test_split(t_data, test_size=0.20)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

vae.fit(x_train,
        x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

encoder = Model(x, z_mu)


# display a 2D manifold of the digits
n = 64


# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)
x_decoded = decoder.predict(z_grid.reshape(64, latent_dim))
x_decoded = x_decoded.reshape(64,25860)



result_data = []
rd = []
for i in range(len(x_decoded)):
    rd = x_decoded[i]
    rd = rd.reshape(20,1293)
    result_data.append(rd)
    
    
result_array = np.array(result_data)

#decoder.save('vae_model_decoder_rs9.h5')
#vae.save('vae_main_model_rs9.h5')
np.save('mfcc_result10'+'.npy',result_array)

#SVG(model_to_dot(vae).create(prog='dot', format='svg'))
print(vae.summary())
