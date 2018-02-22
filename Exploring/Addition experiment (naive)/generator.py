from random import randint
import numpy as np

def getParity(bits):
    odd = 0
    for i,bit in enumerate(bits):
        if bit == "1":
            odd += 1
            odd %= 2

    return odd

def generator(sizeSample, repetition):

    generated = []
    i = 0

    while i < sizeSample:
        int1 = randint(0, 1000)
        int2 = randint(0, 1000)
        added = int1+int2

        result = str(int1) + " " + str(int2)+ " " + str(added)

        if repetition == False:
            if not(result in generated):
                generated.append(result)
                i += 1
        else:
            i += 1

    return generated

def output_data(data):
    with open('data', 'w') as f1:
        for d in data:
            print(d, file=f1)

if __name__ == '__main__':
    generated = generator(1000, False)
    output_data(generated)
