import tensorflow as tf
import tflearn
import numpy as np


def readFiles(fileList):
    dataset = []
    for file in fileList:
        with open('Datasets/' + file) as f:
            data = f.readlines()
        [dataset.append(element) for line in data for element in list(line.strip())]
    return dataset

def preProcess(rawdata):
    characters = sorted(set(rawdata))
    cToIx = {x:i for i, x in enumerate(characters)}
    ixToC = {i:x for i, x in enumerate(characters)}
    idxs = [cToIx[c] for c in rawdata]
    return characters, cToIx, ixToC, idxs

def generateHyperParameters(eta = 0.1, m = 100, seqLength = 25, epochs = 10):
    hp = {'eta': eta, 'm': m, 'seqLength': seqLength, 'epochs': epochs}
    return hp

def generateDatasets(idxs, characters):
    X = np.eye(len(characters))[idxs]
    Y = np.copy(X)
    Y = np.roll(Y, -1, axis = 0)
    return X, Y


def initializeNetwork():
    return 0


rawdata = readFiles(['goblet_book.txt'])
characters, cToIx, ixToC, idxs = preProcess(rawdata)
hp = generateHyperParameters()
X, Y = generateDatasets(idxs, characters)
dataset = tf.data.Dataset.from_tensor_slices(X)

#sequence generation
net = tflearn.input_data(shape=[None,len(rawdata), len(characters)])
net = tflearn.lstm(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, len(characters), activation='softmax')
net = tflearn.regression(net, optimizer = 'adam', loss = 'categorical_crossentropy')
model = tflearn.SequenceGenerator(net, dictionary = cToIx, seq_maxlen = len(rawdata), clip_gradients = 5.0)
X = np.reshape(X, (1, len(rawdata), len(characters)))
model.fit(X, Y)
print(model.generate(10, temperature = 1.0, seq_seed = characters[10]))
#[print(ixToC[np.where(x == 1)[0][0]]) for x in X[0:22]]
#print("\n")
#[print(ixToC[np.where(x == 1)[0][0]]) for x in Y[0:22]]