addpath Functions/;
addpath Datasets/cifar-10-batches-mat/;
addpath Helpfunctions/;

[Xtrain, Ytrain, ytrain, Ntrain] = LoadBatch('data_batch_1.mat');
[Xval, Yval, yval, Nval] = LoadBatch('data_batch_2.mat');
[Xtest, Ytest, ytest, ntest] = LoadBatch('test_batch.mat');
d = rows(Xtrain);
N = Ntrain;
K = rows(Ytrain);
N = 150;
Xtrain = Xtrain(:,1:N);
Ytrain = Ytrain(:,1:N);
ytrain = ytrain(1:N,:);
Ntrain = N;
hiddenNodes = 50;
[Xtrain, mean_of_Xtrain] = Preprocess(Xtrain);
[W, b] = Initialize(K, d, hiddenNodes);
[Wm, bm] = InitializeMomentum(W,b);
n_epochs = 100;
n_batch = 64;
eta = 0.10;
rho = 0.9;
lambda = 0;
J_train = [];
J_val = [];
%%%% Gradient checking procedure
%[grad_b_n, grad_W_n] = ComputeGradsNumSlow(Xtrain, Ytrain, W, b, N, 0, 1e-5);
%for i=1:2
%  disp(GradChecker(grad_W_a{i}, grad_W_n{i}));
%  disp(GradChecker(grad_b_a{i}, grad_b_n{i}));
%endfor
for i=1:n_epochs
  for j=1:N/n_batch
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start:j_end;
    Xbatch = Xtrain(:,inds);
    Ybatch = Ytrain(:,inds);
    [W,b] = MiniBatchGD(Xbatch, Ybatch, eta, W, b, Wm, bm, n_batch,lambda, rho);
  endfor
  costTrain = ComputeCost(Xtrain, Ytrain, W, b, N, lambda);
  accTrain = ComputeAccuracy(Xtrain, ytrain, W, b, N);
  disp('cost: '),disp(costTrain);
  disp('acc: '),disp(accTrain);
  J_train = [J_train costTrain];
  costVal = ComputeCost(Xval, Yval, W, b, Nval, lambda);
  J_val = [J_val costVal];
endfor
disp('test acc: '),disp(ComputeAccuracy(Xtest,ytest, W, b,ntest));
graphics_toolkit gnuplot;
fig = figure();
set(fig, 'visible', 'off');
set(0, 'defaultaxesfontname', 'Helvetica');
hold on;
plot(1:n_epochs, J_train, 'b', 1:n_epochs, J_val, 'y');
title('Cost with lambda = 0, #epochs = 400, #batches = 64, eta = 0.05, 100 examples, Gaussian init');
legend('Training Cost', 'Validation Cost');
print -djpg image.jpg;
