import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, TimeDistributed, Reshape, RepeatVector
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from IPython import embed

def import_data(dataset="data"):
    X = []
    Y = []
    isFirst = True
    bitLength = 0

    f=open(dataset, 'r')
    for line in f.readlines():
        if isFirst:
            bitLength = int(line)
            isFirst = False
        else:
            intLine = [int(s) for s in line.split(' ')]
            X.append(intLine[:bitLength])
            X.append(intLine[bitLength:bitLength*2])
            Y.append(intLine[bitLength*2:][::-1])


    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0],int(Y.shape[1]),1)

    X = np.array(X)
    # X = X.reshape(int(X.shape[0]/2), X.shape[1]*2,1)

    embed()
    X = X.reshape(int(X.shape[0]/2), 2, X.shape[1])
    X = np.array([a.transpose()[::-1] for a in X])

    return X,Y

def convert_to_trainable(examples):
    Y = [list(x) for x in zip(*examples)]
    Y = [np.array(x) for x in Y]
    return Y

def create_RNN_model(input_shape, output_size):
    model = Sequential()
    # model.add(LSTM(10, input_shape=(input_shape[1], input_shape[2]), return_sequences=True, activation="relu"))
    # model.add(TimeDistributed(Dense(2, activation="sigmoid")))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # return model

    model.add(LSTM(100, input_shape=(input_shape[1], input_shape[2]), activation="relu"))
    model.add(RepeatVector(output_size))
    model.add(LSTM(100, return_sequences=True, activation="relu"))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    test_size = 0.1
    epochs = 2000
    b_size = 100

    X, Y = import_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

    classifier = create_RNN_model( X.shape, len(Y[0]))
    classifier.summary()

    classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=2)
    result = classifier.evaluate(X_test, Y_test, batch_size=b_size)


    print("\n\n")
    print(classifier.metrics_names)
    print(result)
