import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, LSTM, TimeDistributed, Reshape
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
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
    X = X.reshape(X.shape[0],X.shape[1],1)
    Y = np.array(Y)

    return X,Y

def create_convolutional_model(seq_length):
    model = Sequential()
    model.add(Conv1D(50, 10, activation='relu', kernel_initializer="glorot_uniform", input_shape=(seq_length[1], 1 )))
    # model.add(Conv1D(64, 1, activation='relu'))
    model.add(MaxPooling1D(2))
    # # model.add(Conv1D(64, 1, activation='relu'))
    model.add(Conv1D(50, 5, activation='relu', kernel_initializer="glorot_uniform"))
    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    test_size = 0.1
    epochs = 100
    b_size = 30

    X, Y = import_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

    # embed()
    classifier = create_convolutional_model(X_train.shape)
    classifier.summary()

    classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=1)
    result = classifier.evaluate(X_test, Y_test, batch_size=b_size)

    print("\n\n")
    print(classifier.metrics_names)
    print(result)
