addpath Functions/
addpath Helpfunctions/
[X, Y, y, N] = LoadBatch('data_batch_1.mat');
d = rows(X);
K = rows(Y);
N = 20;
d = 200;
X = X(1:d,1:N);
Y = Y(:,1:N);
y = y(1:N,:);
layerData = [10,10,10, K];
[W, b] = Initialize(d, layerData, 'gaussi');
[Wm, bm] = InitializeMomentum(W, b);
disp('Displaying Weights');
cellfun(@(x) disp(size(x)), W, 'UniformOutput', false);
cache = EvaluateClassifier(X, W, b);
disp('displaying Cache');
cellfun(@(x) disp(size(x)), cache, 'UniformOutput', false);
disp('displaying cost');
J = ComputeCost(X, Y, W, b, N, 0)
disp('displaying accuracy');
acc = ComputeAccuracy(X, y, W, b, N)
disp('test gradients');
[grad_b, grad_W] = ComputeGradients(X, Y, cache, W, b, N, 0);
[grad_b_num, grad_W_num] = ComputeGradsNumSlow(X, Y, W, b, N, 0, 1e-5);
disp('analytical');
disp(grad_b{2,1});
disp('numerical');
disp(grad_b_num{2,1});
cellfun(@(x, y) GradChecker(x, y), grad_b, grad_b_num, 'UniformOutput', false)
cellfun(@(x, y) GradChecker(x, y), grad_W, grad_W_num, 'UniformOutput', false)
