import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
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
            X.append(intLine[:bitLength*2])
            Y.append(intLine[bitLength*2:])

    return X,Y

def convert_to_trainable(examples):
    Y = [list(x) for x in zip(*examples)]
    Y = [np.array(x) for x in Y]
    return Y

def create_dense_model(input_size, output_size):
    inputs = Input(shape=(input_size,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)

    outputs = []
    for i in range(output_size):
        outputs.append(Dense(1, activation='sigmoid', name='output'+str(i))(x))

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    test_size = 0.1
    epochs = 50
    b_size = 100

    X, Y = import_data()
    X = np.array(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

    Y_train = convert_to_trainable(Y_train)
    Y_test = convert_to_trainable(Y_test)

    classifier = create_dense_model(len(X_train[0]), len(Y[0]))
    classifier.summary()

    classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=2)
    result = classifier.evaluate(X_test, Y_test, batch_size=b_size)

    print("\n\n")
    print(classifier.metrics_names)
    print(result)

    print("\n\n")
    acc = result[len(Y[0])+1:]
    print("Average accuracy of each bit: " + str(sum(acc) / float(len(acc))))
