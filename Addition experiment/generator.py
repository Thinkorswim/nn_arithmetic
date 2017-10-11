import random
import numpy as np

def generator(sizeSample, bitLength, repetition):

    generated = []
    generated.append(bitLength)
    i = 0

    while i < sizeSample:
        intBit1 = random.getrandbits(bitLength)
        intBit1 = bin(intBit1)[2:].zfill(bitLength)

        random.seed()

        intBit2 = random.getrandbits(bitLength)
        intBit2 = bin(intBit2)[2:].zfill(bitLength)

        added = bin(int(intBit1,2) + int(intBit2,2))
        added = added[2:].zfill(bitLength+1)

        result = " ".join(intBit1) + " " + " ".join(intBit2) + " " + " ".join(added)

        if repetition == False:
            if not(result in generated):
                generated.append(result)
                i += 1
        else:
            generated.append(result)
            i += 1

    return generated

def output_data(data):
    with open('data', 'w') as f1:
        for d in data:
            print(d, file=f1)

if __name__ == '__main__':
    generated = generator(100000, 50, True)
    output_data(generated)
