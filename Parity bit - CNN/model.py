import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras import regularizers
from keras import initializers
from keras import optimizers


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

def create_CNN_model(input_size, f, k):
    model = Sequential()
    model.add(Conv1D(f, k, activation='relu', kernel_initializer="glorot_uniform", input_shape=(input_size, 1 )))
    model.add(MaxPooling1D(1))
    model.add(Dropout(0.5))
    model.add(Conv1D(f, 1, activation='relu', kernel_initializer="glorot_uniform"))
    model.add(MaxPooling1D(1))
    model.add(Dropout(0.5))
    model.add(Conv1D(f, 1, activation='relu', kernel_initializer="glorot_uniform"))
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_crossentropy', 'accuracy'])

    return model


if __name__ == '__main__':
    test_size = 0.2
    epochs = 1000
    b_size = 100


    avg_val = np.array([])
    avg_train = np.array([])
    loop = 1

    X, Y = import_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    input_size = len(X_train[0])


    filters = [input_size*2]
    kernels = [input_size]



    print("\nTraining examples: " +  str(X_train.shape[0]))
    print("Test examples: " +  str(X_test.shape[0]))

    for f in filters:
        for k in kernels:
            for i in range(loop):
                classifier = create_CNN_model(len(X_train[0]), f, k)
                classifier.summary()

                history = classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=1)
                result = classifier.evaluate(X_test, Y_test, batch_size=b_size)

                avg_val = np.append(avg_val, result[2])
                avg_train = np.append(avg_train, history.history['acc'][-1])


            print("\n----------------------\n")
            print("Filter: " + str(f))
            print("Kernels: " + str(k))

            print("\nValidation Avg: " + str(np.average(avg_val)))
            print("Train Avg: " + str(np.average(avg_train)))

            print("\n")
            print(avg_val)
            print(avg_train)

            avg_val = np.array([])
            avg_train = np.array([])

    # embed()
