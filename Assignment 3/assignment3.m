addpath Functions/

[X, Y, y, N] = LoadBatch('data_batch_1.mat');
d = rows(X);
K = rows(Y);
layerData = [50, K];
[W, b] = Initialize(d, layerData, 'gaussi');
[Wm, bm] = InitializeMomentum(W, b);
disp('Displaying Weights');
cellfun(@(x) disp(size(x)), W, 'UniformOutput', false);
cache = EvaluateClassifier(X, W, b);
disp('displaying Cache');
cellfun(@(x) disp(size(x)), cache, 'UniformOutput', false);
