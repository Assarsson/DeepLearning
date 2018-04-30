addpath Functions/
addpath Helpfunctions/

[X, Y, y, N] = LoadBatch('data_batch_1.mat');
d = rows(X);
K = rows(Y);

%% change parameters here (i.e. dimensionality of operations)
d = 150;
N = 20;
%Reshape the data
X = X(1:d,1:N);
Y = Y(:,1:N);
y = y(1:N,:);

%initialize network parameters
layerData = [50, K]; %The same as in assignment 2
lambda = 0;
[W, b] = Initialize(d, layerData, 'gaussi');
cache = EvaluateClassifier(X, W, b);
[dba, dWa] = ComputeGradients(X, Y, cache, W, b, N, lambda);
[dbn, dWn] = ComputeGradsNumSlow(X, Y, W, b, N, lambda, 1e-5);

disp('differences in the bias gradients: ');
cellfun(@(x, y) disp(GradChecker(x, y)), dba, dbn, 'UniformOutput', false);
disp('difference in the weight gradients: ');
cellfun(@(x, y) disp(GradChecker(x, y)), dWa, dWn, 'UniformOutput', false);
