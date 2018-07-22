
addpath Functions/
addpath Helpfunctions/
fileName = 'goblet_book.txt';
bookData = LoadBatch(fileName);
disp('loaded book data');
hp = GenerateHyperParameters();
disp('generated hyper parameters');
[bookChars, cToIx, ixToC, K] = Preprocess(bookData);
disp('finished preprocessing data and creating index maps');
hp.K = K;
disp('Initializing parameters and one-hot-encoding dataset...');
[RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx, ixToC, testing = false);
disp('initialized network and parameters');
disp('running MiniBatch Gradient Descent');
RNN = MiniBatchGD(X, RNN, hp);
