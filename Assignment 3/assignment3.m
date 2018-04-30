addpath Functions/
addpath Helpfunctions/
[Xtrain, Ytrain, ytrain, Ntrain] = LoadAll({'data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat'});
[Xval, Yval, yval, Nval] = LoadBatch('data_batch_5.mat');
[Xtest, Ytest, ytest, ntest] = LoadBatch('test_batch.mat');
ytrain = ytrain';
[Xtrain, mean_of_Xtrain] = Preprocess(Xtrain);
d = rows(Xtrain);
K = rows(Ytrain);
layerData = [50, K];
[W, b] = Initialize(d, layerData, 'xavier');
[Wm, bm] = InitializeMomentum(W, b);
n_epochs = 10;
n_batch = 64;
eta = 0.020815;
lambda = 0.00023554;
N = Ntrain;
rho = 0.999;
for epoch = 1:n_epochs
  for j = 1:N/n_batch
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start:j_end;
    Xbatch = Xtrain(:,inds);
    Ybatch = Ytrain(:,inds);
    [W,b,Wm,bm] = MiniBatchGD(Xbatch, Ybatch, eta, W, b, Wm, bm, n_batch,lambda, rho);
  endfor
  costTrain = ComputeCost(Xtrain, Ytrain, W, b, N, lambda);
  accTrain = ComputeAccuracy(Xtrain, ytrain, W, b, N);
  disp('Cost at current epoch: '),disp(costTrain);
  disp('Accuracy at current epoch: '),disp(accTrain);
endfor
