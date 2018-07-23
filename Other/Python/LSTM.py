import tensorflow as tf
import tflearn
import numpy as np
from tflearn.data_utils import *


def readFiles(fileName, hp):
    X, Y, idxs = textfile_to_semi_redundant_sequences("Datasets/"+fileName, seq_maxlen = hp['seqLength'], redun_step = 3)
    return X, Y, idxs
def generateHyperParameters(eta = 0.1, m = 100, seqLength = 25, epochs = 10):
    hp = {'eta': eta, 'm': m, 'seqLength': seqLength, 'epochs': epochs}
    return hp
print("reading in files...")
fileName = 'goblet_book.txt'
hp = generateHyperParameters()
X, Y, idxs = readFiles(fileName, hp)
#sequence generation

print("building tensorflow model...")
net = tflearn.input_data(shape=[None,hp['seqLength'], len(idxs)])
print("added input layer")
net = tflearn.lstm(net, 128)
print("added lstm layer")
net = tflearn.dropout(net, 0.3)
print("added dropout")
net = tflearn.lstm(net, 128)
print("added second lstm layer")
net = tflearn.dropout(net, 0.3)
net = tflearn.fully_connected(net,   len(idxs), activation='softmax')
print("added fully connected softmax")
net = tflearn.regression(net, optimizer = 'adam', loss = 'categorical_crossentropy')
print("added ADAM optimized cross entropy loss")
model = tflearn.SequenceGenerator(net, dictionary = idxs, seq_maxlen = hp['seqLength'], clip_gradients = 5.0)
print("created sequence generator")
print("model built!")
print("fitting model...")
model.fit(X, Y, n_epoch = 20)
print("model fitted!")
print("generating sequence...")
seed = random_sequence_from_textfile("Datasets/"+fileName, hp['seqLength'])
print("sequence with temperature 1.0")
print(model.generate(200, temperature = 1.0, seq_seed = seed))
print("sequence with temperature 0.5")
print(model.generate(200, temperature = 0.5, seq_seed = seed))
#[print(ixToC[np.where(x == 1)[0][0]]) for x in X[0:22]]
#print("\n")
#[print(ixToC[np.where(x == 1)[0][0]]) for x in Y[0:22]]
