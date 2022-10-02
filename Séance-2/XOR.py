import tensorflow as tf
from tensorflow import keras 
import numpy as np 

from keras import Sequential 
from keras.layers import Dense, Activation 
from keras.optimizers import SGD

X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]) 
y = np.array([[0.],[1.],[1.],[0.]])

model = Sequential() 
model.add(Dense(8, input_dim=2)) 
model.add(Activation('tanh')) 
model.add(Dense(1)) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd)
model.fit(X, y, verbose=1, batch_size=1, epochs=1000) 

print(model.predict(X))