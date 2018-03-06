import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, LSTM, TimeDistributed, Reshape
from keras import regularizers, initializers

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
    X = X.reshape(X.shape[0], 1, X.shape[1])
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0], 1)

    return X,Y

def create_RNN_model(input_shape):
    model = Sequential()
    model.add(LSTM(input_shape[2]+1, input_shape=(input_shape[1], input_shape[2]), activation="selu", kernel_initializer=initializers.Orthogonal(gain=1.0, seed=0), return_sequences=False))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['binary_crossentropy','accuracy'])

    return model


if __name__ == '__main__':
    test_size = 0.2
    epochs = 1000
    b_size = 100

    avg_val = np.array([])
    avg_train = np.array([])
    loop = 1

    X, Y = import_data()

    # embed()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

    for i in range(loop):
        classifier = create_RNN_model(X_train.shape)
        classifier.summary()

        history = classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=1)
        result = classifier.evaluate(X_test, Y_test, batch_size=b_size)

        avg_val = np.append(avg_val, result[2])
        avg_train = np.append(avg_train, history.history['acc'][-1])


    print("\n----------------------\n")

    print("\nValidation Avg: " + str(np.average(avg_val)))
    print("Train Avg: " + str(np.average(avg_train)))

    print("\n")
    print(avg_val)
    print(avg_train)

    avg_val = np.array([])
    avg_train = np.array([])

    embed()
