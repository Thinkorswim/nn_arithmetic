import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
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

    return X,Y

def create_dense_model(input_size):
    model = Sequential()
    model.add(Dense(1, activation="linear", input_shape=(input_size,)))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])
    return model


if __name__ == '__main__':
    test_size = 0.1
    epochs = 200
    b_size = 10

    X, Y = import_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

    classifier = create_dense_model(len(X_train[0]))
    classifier.summary()

    classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=1)
    result = classifier.evaluate(X_test, Y_test, batch_size=b_size)

    print("\n\n")
    print(classifier.metrics_names)
    print(result)
