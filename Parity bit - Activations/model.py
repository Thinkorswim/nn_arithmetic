import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from keras.layers import Input
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

    if input_size % 2 == 0:
        model.add(Dense(int(input_size/2 + 1), activation="selu", input_shape=(input_size,)))
    else:
        model.add(Dense(int((input_size+1)/2), activation="selu", input_shape=(input_size,)))

    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])
    return model


if __name__ == '__main__':
    test_size = 0.2
    epochs = 250
    b_size = 100

    avg_val = np.array([])
    avg_train = np.array([])
    loop = 10

    X, Y = import_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    print("\nTraining examples: " +  str(X_train.shape[0]))
    print("Test examples: " +  str(X_test.shape[0]))

    for i in range(loop):
        classifier = create_dense_model(len(X_train[0]))
        #classifier.summary()

        history = classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=1)
        result = classifier.evaluate(X_test, Y_test, batch_size=b_size)

        avg_val = np.append(avg_val, result[2])
        avg_train = np.append(avg_train, history.history['acc'][-1])


    print("\nValidation Avg: " + str(np.average(avg_val)))
    print("Train Avg: " + str(np.average(avg_train)))

    print("\n")
    print(avg_val)
    print(avg_train)

    print("hard_sigmoid")