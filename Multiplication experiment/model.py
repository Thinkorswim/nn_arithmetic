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
    input1 = Input(shape=(input_size,))
    input2 = Input(shape=(input_size,))

    input = keras.layers.concatenate([input1, input2])
    x = Dense(64, activation='selu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(input)
    x = Dense(64, activation='selu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='selu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='selu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)

    #

    outputs = []
    inputs=[input1, input2]
    for i in range(output_size):
        outputs.append(Dense(1, activation='sigmoid', name='output'+str(i))(x))

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    test_size = 0.1
    epochs = 1000
    b_size = 10

    X, Y = import_data()
    X = np.array(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

    Y_train = convert_to_trainable(Y_train)
    Y_test = convert_to_trainable(Y_test)

    # embed()
    X_train = X_train.reshape(X_train.shape[0], 2, int(X_train.shape[1]/2))
    X_train = [X_train[:,0,:], X_train[:,1,:]]

    X_test = X_test.reshape(X_test.shape[0], 2, int(X_test.shape[1]/2))
    X_test = [X_test[:,0,:], X_test[:,1,:]]
    X_test = np.array(X_test)


    classifier = create_dense_model(int(X.shape[1]/2), len(Y[0]))
    classifier.summary()

    classifier.fit(X_train, Y_train, epochs=epochs, batch_size=b_size, verbose=2)
    result = classifier.evaluate([X_test[0,:,:], X_test[1,:,:]], Y_test, batch_size=b_size)

    # embed()
    correctCount = 0
    for i in range(X_test.shape[1]):
        r = np.around(classifier.predict([X_test[0,i,:].reshape(1,-1),X_test[1,i,:].reshape(1,-1)], batch_size=b_size)).reshape(len(Y[0]),1)
        # print(result.tolist(), np.array(Y_test)[:,i].tolist())
        if r.tolist() == np.array(Y_test)[:,i].tolist():
           correctCount+=1

    print("\n\nCorrect: " + str(correctCount))
    print("Total: " + str(X_test.shape[1]))
    print("Percentage: " + str(correctCount/X_test.shape[1]))

    embed()
    print("\n\n")
    # print(classifier.metrics_names[len(Y[0])+1:])
    print(result[len(Y[0])+1:])

    print("\n\n")
    acc = result[len(Y[0])+1:]
    print("Average accuracy of each bit: " + str(sum(acc) / float(len(acc))))
