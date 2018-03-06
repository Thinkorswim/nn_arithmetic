import random
import numpy as np

def getParity(bits):
    odd = 0
    for i,bit in enumerate(bits):
        if bit == "1":
            odd += 1
            odd %= 2

    return odd

def generator(sizeSample, bitLength, repetition):

    generated = []
    i = 0

    while i < sizeSample:
        intBit = random.getrandbits(bitLength)
        bits = bin(intBit)[2:].zfill(bitLength)

        bitString = " ".join(bits) + " " +str(getParity(bits));

        if repetition == False:
            if not(bitString in generated):
                generated.append(bitString)
                i += 1
        else:
            generated.append(bitString)
            i += 1

    return generated

def output_data(data):
    with open('data', 'w') as f1:
        for d in data:
            print(d, file=f1)

if __name__ == '__main__':
    generated = generator(1024, 10, False)
    output_data(generated)
