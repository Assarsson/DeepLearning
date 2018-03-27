clear;

[Xtrain, Ytrain, ytrain, Ntrain] = LoadBatch('data_batch_1.mat');
[Xval, Yval, yval, Nval] = LoadBatch('data_batch_2.mat');
[Xtest, Ytest, ytest, Ntest] = LoadBatch('test_batch.mat');
K = max(ytrain);
disp(K);
d = rows(Xtrain);
disp(d);
N = Ntrain;
disp(N);
lambda = 0;
GDparams.n_batch = 100;
GDparams.eta = .01;
GDparams.n_epochs = 40;
[W,b] = Initialize(K,d);
[W,b] = MiniBatchGD(Xtrain, Ytrain, ytrain, GDparams, W, b, N,lambda);
disp("Test accuracy: "),disp(ComputeAccuracy(Xtest, ytest, W, b, N));
for i=1:K
  im = reshape(W(i, :), 32, 32, 3);
  s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
  s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'size', [1,K]);
pause(50);
