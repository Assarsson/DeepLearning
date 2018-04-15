addpath Functions/;
addpath Datasets/cifar-10-batches-mat/;
addpath Helpfunctions/;

[Xtrain, Ytrain, ytrain, Ntrain] = LoadBatch('data_batch_1.mat');
[Xval, Yval, yval, Nval] = LoadBatch('data_batch_1.mat');
[Xtest, Ytest, ytest, ntest] = LoadBatch('test_batch.mat');

d = rows(Xtrain);
N = Ntrain;
K = rows(Ytrain);
hiddenNodes = 50;
Xtrain = Xtrain(1:20,1:20);
Ytrain = Ytrain(:,1:20);
N = 20;
d = 20;
[Xtrain, mean_of_Xtrain] = Preprocess(Xtrain);
[W, b] = Initialize(K, d, hiddenNodes);
cache = EvaluateClassifier(Xtrain, W, b);
[grad_b_a, grad_W_a] = ComputeGradients(Xtrain, Ytrain, cache, W, b, N, 0);

%%%% Gradient checking procedure
% [grad_b_n, grad_W_n] = ComputeGradsNum(Xtrain, Ytrain, W, b, N, 0, 1e-6);
% for i=1:2
%  disp(GradChecker(grad_W_a{i}, grad_W_n{i}));
%  disp(GradChecker(grad_b_a{i}, grad_b_n{i}));
%endfor
