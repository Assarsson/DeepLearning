addpath BatchNormalize/
addpath Helpfunctions/
[X, Y, y, N] = LoadBatch('data_batch_1.mat');
d = rows(X);
K = rows(Y);

%% change parameters here (i.e. dimensionality of operations)
d = 20;
N = 2;
%Reshape the data
X = X(1:d,1:N);
Y = Y(:,1:N);
y = y(1:N,:);

%initialize network parameters
layerData = [50,30, K]; %The same as in assignment 2
lambda = 0;
[W, b] = Initialize(d, layerData, 'gaussi');
[P, S,Shat, H, mus, vs] = EvaluateClassifier(X, W, b);
[grad_b, grad_W] = ComputeGradients(X, Y, P,S,Shat,H, mus, vs, W, b, N, lambda);
[grad_b_n, grad_W_n] = ComputeGradsNumSlow(X, Y, W, b, N, lambda, 1e-5);

disp('differences in the bias gradients: ');
cellfun(@(x, y) disp(GradChecker(x, y)), grad_b, grad_b_n, 'UniformOutput', false);
disp('differences in the weight gradients: ');
cellfun(@(x, y) disp(GradChecker(x, y)), grad_W, grad_W_n, 'UniformOutput', false);

disp('size of BNcache parameters: ');
cellfun(@(x) disp(size(x)), mus, 'UniformOutput', false);
disp('size of b');
cellfun(@(x) disp(size(x)), grad_b, 'UniformOutput', false);
disp('size of W');
cellfun(@(x) disp(size(x)), grad_W, 'UniformOutput', false);
