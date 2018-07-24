import tensorflow as tf
import tflearn
import numpy as np
from tflearn.data_utils import *
import argparse
import sys


def readFiles(fileName, hp, redun_step = 3):
    try:
        print("reading in files...")
        X, Y, idxs = textfile_to_semi_redundant_sequences("Datasets/"+fileName, seq_maxlen = hp['seqLength'], redun_step = redun_step)
        seed = random_sequence_from_textfile("Datasets/"+fileName, hp['seqLength'])
        return X, Y, idxs, seed
    except FileNotFoundError:
        print("please choose a file that exists in the Dataset-folder")
        sys.exit(1)
def generateHyperParameters(seqLength = 25, epochs = 10, genlen = 200, eta = 0.001, m = 128):
    hp = {'eta': eta, 'm': m, 'genlen': genlen, 'seqLength': seqLength}
    return hp
def buildModel(hp, idxs, layers):
    print("building tensorflow model...")
    net = tflearn.input_data(shape=[None,hp['seqLength'], len(idxs)])
    print("added input layer")
    for layer in range(layers-1):
        net = tflearn.lstm(net, hp['m'], return_seq = True)
        print("added lstm layer")
        net = tflearn.dropout(net, 0.3)
        print("added dropout")
    net = tflearn.lstm(net, hp['m'])
    print("added final lstm layer")
    net = tflearn.dropout(net, 0.3)
    net = tflearn.fully_connected(net,   len(idxs), activation='softmax')
    print("added fully connected softmax")
    net = tflearn.regression(net, optimizer = 'adam', loss = 'categorical_crossentropy', learning_rate = hp['eta'])
    print("added ADAM optimized cross entropy loss")
    model = tflearn.SequenceGenerator(net, dictionary = idxs, seq_maxlen = hp['seqLength'], clip_gradients = 5.0)
    print("created sequence generator")
    print("model built!")
    return model
def parseInputArguments():
    parser = argparse.ArgumentParser(description = "Process the integer parameters the LSTM-model")
    parser.add_argument('--epochs', type = int, dest = 'epochs', default = 10, help = "defines the number of epochs")
    parser.add_argument('--filename', type = str, dest = 'fileName', default = "goblet_book.txt", help = "define a file in the Datasets folder")
    parser.add_argument('--genlen', type = int, dest = 'genlen', default = 200, help = "the length of the generated message")
    parser.add_argument('--seqlength', type = int, dest = 'seqLength', default = 25, help = 'training sequence length')
    parser.add_argument('--learningrate', type = float, dest = 'eta', default = 0.001, help = 'Learning rate for the model')
    parser.add_argument('--load', type = bool, dest = 'load', default = False, help = 'bool to indicate if loading a model or training')
    parser.add_argument('--layers', type = int, dest = 'layers', default = 2, help = 'the number of layers to add to the LSTM-model')
    arguments = parser.parse_args()
    return arguments
def sequenceGenerator(arguments, seed, model):
    print("generating sequence...")
    print("sequence with temperature 1.0")
    print(model.generate(arguments.genlen, temperature = 1.1, seq_seed = seed))
    print("sequence with temperature 0.5")
    print(model.generate(arguments.genlen, temperature = 0.5, seq_seed = seed))

if __name__ == "__main__":
    arguments = parseInputArguments()
    hp = generateHyperParameters(arguments.seqLength, arguments.epochs, arguments.genlen, arguments.eta)
    X, Y, idxs, seed = readFiles(arguments.fileName, hp)
    model = buildModel(hp, idxs, arguments.layers)
    if arguments.load == False:
        print("fitting model...")
        model.fit(X, Y, n_epoch = arguments.epochs)
        print("model fitted!")
        model.save("{}_{}_{}.tfl".format(arguments.fileName, arguments.seqLength, arguments.epochs))
    else:
        print("loading model...")
        model.load("Results/{}_{}_{}.tfl".format(arguments.fileName, arguments.seqLength, arguments.epochs), weights_only=True)
    sequenceGenerator(arguments, seed, model)
