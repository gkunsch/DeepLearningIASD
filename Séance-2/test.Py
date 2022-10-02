#not made to be run, but as an example

import tensorflow as tf
from tensorflow import keras 
import numpy as np 

from keras import Sequential 
model = Sequential()

#libraries for defining the layers and the libraries for the training algorithms:
from keras.layers import Dense, Activation 
from keras.optimizers import SGD

# after the first layer, you don't need to specify # the size of the input anymore:
model = Sequential()
model.add(Dense(units=64, input_dim=100))
model.add(Activation('tanh')) 
model.add(Dense(units=10)) 
model.add(Activation('softmax'))

#Input and labels are represented as arrays:
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]) 
y = np.array([[0.],[1.],[1.],[0.]])

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)
model.fit(X, y, verbose=1, batch_size=1, epochs=1000)
print(model.predict(X))