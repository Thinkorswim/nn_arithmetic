import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, LSTM, TimeDistributed, Reshape
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from IPython import embed



def import_data(dataset="data"):
    X = []
    Y = []

    f=open(dataset, 'r')
    for line in f.readlines():
        intLine = [int(s) for s in line.split(' ')]
        X.append(intLine[:-1])
        Y.append(intLine[-1:])

    X = np.array(X)
    X = X.reshape(X.shape[0],1 ,X.shape[1])
    Y = np.array(Y)

    return X,Y

def create_RNN_model(input_shape):
    model = Sequential()
    model.add(LSTM(30, input_shape=(input_shape[1], input_shape[2]), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    test_size = 0.1
    epochs = 1000
    b_size = 100

    X, Y = import_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

    embed()
    classifier = create_RNN_model(X_train.shape)
    classifier.summary()

    classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=1)
    result = classifier.evaluate(X_test, Y_test, batch_size=b_size)

    print("\n\n")
    print(classifier.metrics_names)
    print(result)
