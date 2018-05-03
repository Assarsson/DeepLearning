addpath Functions/
addpath Helpfunctions/
[Xtrain, Ytrain, ytrain, Ntrain] = LoadAll({'data_batch_1.mat'});
[Xval, Yval, yval, Nval] = LoadBatch('data_batch_5.mat');
[Xtest, Ytest, ytest, ntest] = LoadBatch('test_batch.mat');
ytrain = ytrain';
[Xtrain, mean_of_Xtrain] = Preprocess(Xtrain);
d = rows(Xtrain);
K = rows(Ytrain);
layerData = [50, 30, 30, K];
[W, b] = Initialize(d, layerData, 'gaussi');
n_epochs = 130;
n_batch = 512;
lambda = 0.000056;
eta = 0.017260;
N = Ntrain;
rho = 0.999;
J_train = [];
J_val = [];
tic;
message = ['Initializing training with ' num2str(N) ' examples and ' num2str(n_epochs) ' epochs...'];
disp(message);
for epoch = 1:n_epochs
  [Wm, bm] = InitializeMomentum(W, b);
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
  costMessage = ['Cost at epoch ' num2str(epoch) ': ' num2str(costTrain)];
  disp(costMessage);
  accMessage = ['Accuracy at epoch ' num2str(epoch) ': ' num2str(accTrain)];
  disp(accMessage);
  if epoch == 1
    timeMessage = ['Estimated total run time in minutes: ' num2str(round(toc*n_epochs/60))];
    disp(timeMessage);
  endif
  J_train = [J_train costTrain];
  costVal = ComputeCost(Xval-repmat(mean_of_Xtrain,[1,size(Xval,2)]), Yval, W, b, Nval, lambda);
  J_val = [J_val costVal];
endfor

graphics_toolkit gnuplot;
fig = figure();
set(fig, 'visible', 'off');
set(0, 'defaultaxesfontname', 'Helvetica');
hold on;
titleText = ['Cost with layers = ' num2str(length(layerData)) ' lambda = ' num2str(lambda) ' #epochs = ' num2str(n_epochs) ' #batches = ' num2str(n_batch) ' eta = ' num2str(eta) ' examples = ' num2str(N)];
plot(1:n_epochs, J_train, 'b', 1:n_epochs, J_val, 'y');
title(titleText);
imageName = ['eta' num2str(eta) '+lambda' num2str(lambda) '.jpg'];
legend('Training Cost', 'Validation Cost');
print(imageName);
