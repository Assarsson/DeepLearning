addpath BatchNormalize/
addpath Helpfunctions/
addpath Datasets/cifar-10-batches-mat/
[Xtrain, Ytrain, ytrain, Ntrain] = LoadAll({'data_batch_1.mat'});
[Xval, Yval, yval, Nval] = LoadBatch('data_batch_5.mat');
[Xtest, Ytest, ytest, Ntest] = LoadBatch('test_batch.mat');
ytrain = ytrain';
[Xtrain, mean_of_Xtrain] = Preprocess(Xtrain);
d = rows(Xtrain);
K = rows(Ytrain);
layerData = [50, K];
[W, b] = Initialize(d, layerData, 'gaussi');
n_epochs = 20;
n_batch = 512;
no_etas = 6;
no_lambdas = 1;
etas = [0.05];
decayRate = 0.9;
lambdas = [1e-6];
titleText = ['searching over a total of ' num2str(no_etas*no_lambdas) ' parameters.'];
disp(titleText);
%lambda = 0.000056;
%eta = 0.017260;
alph = 0.99;
N = Ntrain;
rho = 0.999;
J_train = [];
J_val = [];
tic;
message = ['Initializing ' num2str(length(layerData)) '-layer training with ' num2str(N) ' examples and ' num2str(n_epochs) ' epochs...'];
disp(message);
bestAccuracies = [];
for eta = etas
  disp(eta);
  for lambda = lambdas
    [W, b] = Initialize(d, layerData, 'gaussi');
    mav = cell(size(layerData));
    vav = cell(size(layerData));
    J_train = [];
    J_val = [];
    tempAccuracies = [];
    tic;
    valCost = 10000;
    iterator = 0;
    for epoch = 1:n_epochs
      [Wm, bm] = InitializeMomentum(W, b);
      for j = 1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = Xtrain(:,inds);
        Ybatch = Ytrain(:,inds);
        [W,b,Wm,bm,mav,vav] = MiniBatchGD(Xbatch, Ybatch, eta, W, b, Wm, bm, n_batch,lambda, rho, mav, vav, alph);
      endfor
      eta = eta*decayRate;
      costTrain = ComputeCost(Xtrain, Ytrain, W, b, N, lambda, mav, vav);
      accTrain = ComputeAccuracy(Xtrain, ytrain, W, b, N, mav, vav);
      if epoch == 1
        timeMessage = ['Estimated total run time in minutes: ' num2str(round(toc*n_epochs/60))];
        disp(timeMessage);
      endif
      costMessage = ['Cost at epoch ' num2str(epoch) ': ' num2str(costTrain)];
      disp(costMessage);
      accMessage = ['Accuracy at epoch ' num2str(epoch) ': ' num2str(accTrain)];
      disp(accMessage);
      J_train = [J_train costTrain];
      costVal = ComputeCost(Xval-repmat(mean_of_Xtrain,[1,size(Xval,2)]), Yval, W, b, Nval, lambda, mav, vav);
      costValMessage = ['Validation cost at epoch ' num2str(epoch) ': ' num2str(costVal)];
      accVal = ComputeAccuracy(Xval-repmat(mean_of_Xtrain,[1,size(Xval,2)]), yval, W, b, N, mav, vav);
      costAccMessage = ['Validation accuracy at epoch ' num2str(epoch) ': ' num2str(accVal)];
      disp(costValMessage);
      disp(costAccMessage);
      J_val = [J_val costVal];
    endfor
    tempAccuracies = [tempAccuracies ComputeAccuracy(Xtest-repmat(mean_of_Xtrain,[1,size(Xtest,2)]),ytest, W, b,Ntest, mav, vav)];
    bestAccuracies = [bestAccuracies ['accuracy: ' num2str(max(tempAccuracies)) ' for lambda ' num2str(lambda) ' and eta ' num2str(eta) char(10)]];
    graphics_toolkit gnuplot;
    finalAcc = ['Final model accuracy is: ' num2str(ComputeAccuracy(Xtest-repmat(mean_of_Xtrain,[1,size(Xtest,2)]), ytest, W, b, Ntest, mav, vav)*100) ' %'];
    disp(finalAcc);
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
  endfor
endfor
save bestAccuracies.txt bestAccuracies;
