# -- files io --
import pandas as pd
import pickle
import scipy.io as sio
import numpy as np

# -- modeling --
from keras.preprocessing import sequence
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.layers.recurrent import LSTM
#from keras import optimizer

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# -- visualization --
from visualize import * 

WINDOW = 24

#data = sio.loadmat('lstm_xy.mat')
data = pickle.load(open('lstm_xy.p','r'))

X_train = np.array(data['X_train'])
#print X_train[0]
y_train = np.array(data['y_train'])
X_test = np.array(data['X_test'])
y_test = np.array(data['y_test'])


#X_train = sequence.pad_sequences(X_train, maxlen=WINDOW)
#X_test = sequence.pad_sequences(X_test, maxlen=WINDOW)


in_neurons = X_train.shape[2]
out_neurons = 1
hidden_neurons = 50

model = Sequential()


model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=(None, in_neurons)))
model.add(Dropout(0.1))
model.add(Dense(20, input_dim=hidden_neurons, activation = 'relu'))  
model.add(Dense(out_neurons, input_dim=20 , activation = 'sigmoid'))  
#sgd = optimizer.
model.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy','binary_crossentropy'])


model.summary()
class_weight = {0:1, 1:5}

model.fit(X_train, y_train, batch_size=50, epochs=100, class_weight = class_weight) #, validation_split = 0.2)

y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
print(len(y_train))
print(sum(y_train))
print(sum(y_pred))
print(sum(y_test))
print("\nTest precision: %.2f%%	Test reacall: %.2f%%	" % (precision_score(y_test,y_pred)*100,recall_score(y_test,y_pred)*100))

scores = model.evaluate(X_test, y_test, batch_size=50)
print("\nTest Loss: %.2f%%		Test Accuracy: %.2f%%" % (scores[0]*100, scores[1]*100))

y_pred = model.predict(X_test, batch_size = 250)
plot_roc(y_test, y_pred)

