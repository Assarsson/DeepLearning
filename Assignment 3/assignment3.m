addpath Functions/

[X, Y, y, N] = LoadBatch('data_batch_1.mat');
d = rows(X);
K = rows(Y);
N = 10;
X = X(:,1:N);
Y = Y(:,1:N);
y = y(1:N,:);
layerData = [50, K];
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
