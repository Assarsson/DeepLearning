import tensorflow as tf
import tflearn
import numpy as np
from tflearn.data_utils import *


def readFiles(fileName, hp):
    X, Y, idxs = textfile_to_semi_redundant_sequences("Datasets/"+fileName, seq_maxlen = hp['seqLength'], redun_step = 3)
    return X, Y, idxs
def preProcess(rawdata):
    characters = sorted(set(rawdata))
    cToIx = {x:i for i, x in enumerate(characters)}
    ixToC = {i:x for i, x in enumerate(characters)}
    idxs = [cToIx[c] for c in rawdata]
    return characters, cToIx, ixToC, idxs

def generateHyperParameters(eta = 0.1, m = 100, seqLength = 25, epochs = 10):
    hp = {'eta': eta, 'm': m, 'seqLength': seqLength, 'epochs': epochs}
    return hp

def generateDatasets(idxs, characters, seqLength):
    X = np.eye(len(characters))[idxs]
    X = X[0:np.shape(X)[0]-len(idxs)%seqLength]
    X = np.reshape(X, (len(idxs)//seqLength, seqLength, len(characters)))
    Y = np.copy(X)
    Y = np.roll(Y, -1, axis = 0)
    return X, Y


def initializeNetwork():
    return 0

print("reading in files...")
fileName = 'goblet_book.txt'
hp = generateHyperParameters()
X, Y, idxs = readFiles(fileName, hp)
#sequence generation

print("building tensorflow model...")
net = tflearn.input_data(shape=[None,hp['seqLength'], len(idxs)])
print("added input layer")
net = tflearn.lstm(net, 64)
print("added lstm layer")
net = tflearn.dropout(net, 0.5)
print("added dropout")
net = tflearn.fully_connected(net,   len(idxs), activation='softmax')
print("added fully connected softmax")
net = tflearn.regression(net, optimizer = 'adam', loss = 'categorical_crossentropy')
print("added ADAM optimized cross entropy loss")
model = tflearn.SequenceGenerator(net, dictionary = idxs, seq_maxlen = hp['seqLength'], clip_gradients = 5.0)
print("created sequence generator")
print("model built!")
print("fitting model...")
model.fit(X, Y)
print("model fitted!")
print("generating sequence...")
seed = random_sequence_from_textfile("Datasets/"+fileName, hp['seqLength'])
print(model.generate(10, temperature = 1.0, seq_seed = seed))
#[print(ixToC[np.where(x == 1)[0][0]]) for x in X[0:22]]
#print("\n")
#[print(ixToC[np.where(x == 1)[0][0]]) for x in Y[0:22]]
