import tensorflow as tf
import tflearn
import numpy as np
from tflearn.data_utils import *
import argparse
import sys


def readFiles(fileName, hp):
    try:
        print("reading in files...")
        X, Y, idxs = textfile_to_semi_redundant_sequences("Datasets/"+fileName, seq_maxlen = hp['seqLength'], redun_step = 3)
        seed = random_sequence_from_textfile("Datasets/"+fileName, hp['seqLength'])
        return X, Y, idxs, seed
    except FileNotFoundError:
        print("please choose a file that exists in the Dataset-folder")
        sys.exit(1)
def generateHyperParameters(seqLength = 25, epochs = 10, genlen = 200, eta = 0.01, m = 100):
    hp = {'eta': eta, 'm': m, 'genlen': genlen}
    return hp
def buildModel(hp, idxs):
    print("building tensorflow model...")
    net = tflearn.input_data(shape=[None,hp['seqLength'], len(idxs)])
    print("added input layer")
    net = tflearn.lstm(net, 128, return_seq = True)
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
    return model
def parseFunction():
    parser = argparse.ArgumentParser(description = "Process the integer parameters the LSTM-model")
    parser.add_argument('--epochs', type = int, dest = 'epochs', default = 10, help = "defines the number of epochs")
    parser.add_argument('--filename', type = str, dest = 'fileName', default = "goblet_book.txt", help = "define a file in the Datasets folder")
    parser.add_argument('--genlen', type = int, dest = 'genlen', default = 200, help = "the length of the generated message")
    parser.add_argument('--seqlength', type = int, dest = 'seqLength', default = 25, help = 'training sequence length')
    arguments = parser.parse_args()
    return arguments
def sequenceGenerator(arguments, seed):
    print("generating sequence...")
    print("sequence with temperature 1.0")
    print(model.generate(arguments.genlen, temperature = 1.1, seq_seed = seed))
    print("sequence with temperature 0.5")
    print(model.generate(arguments.genlen, temperature = 0.5, seq_seed = seed))

if __name__ == "__main__":
    arguments = parseFunction()
    hp = generateHyperParameters(arguments.seqLength, arguments.epochs, arguments.genlen)
    X, Y, idxs, seed = readFiles(arguments.fileName, hp)
    model = buildModel(hp, idxs)
    print("fitting model...")
    model.fit(X, Y, n_epoch = arguments.epochs)
    print("model fitted!")
    sequenceGenerator(arguments, seed)
    model.save("Results/LSTM for {}.tfl".format(arguments.fileName))
