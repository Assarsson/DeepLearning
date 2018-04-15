addpath Functions/;
addpath Datasets/cifar-10-batches-mat/;

[Xtrain, Ytrain, ytrain, Ntrain] = LoadBatch('data_batch_1.mat');
[Xval, Yval, yval, Nval] = LoadBatch('data_batch_1.mat');
[Xtest, Ytest, ytest, ntest] = LoadBatch('test_batch.mat');
d = rows(Xtrain);
N = Ntrain;
K = rows(Ytrain);
hiddenNodes = 50;
[Xtrain, mean_of_Xtrain] = Preprocess(Xtrain);
Theta = Initialize(K, d, hiddenNodes);
