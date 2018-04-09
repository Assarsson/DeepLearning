## Information: This program follows the instruction for lab 1 in the Masters Course
## of Deep Learning at the Royal Institute of Technology. It is a multi-class one-layer
## neural network trained with mini-batch gradient descent on both a cross-entropy-loss
## against a softmax output, as well as a SVM multi-class loss.
## Author: Fabian Assarsson & Madeleine Gartz
## Date: 2018-03-25
## TODO: move all disp-commands into a separate debugging function to clean up code
## and include the actual mini-batch GD-algorithm to finish the minireq of lab.
clear;
% First we start with looking at some of the images from the dataset:
addpath Datasets/cifar-10-batches-mat/; % adds our path to data
% A = load('data_batch_1.mat'); % loads a saved datafile with examples as rows in A
% I = reshape(A.data', 32, 32, 3, 10000); % We reshape it into a tensor for the montage-function
% I = permute(I, [2, 1, 3, 4]); % We permute the first and second axis
% montage(I(:,:,:,1:500), 'Size', [5,5]); % We display it with our helper function

% Exercise 1
% Given: W = (K*d), x = (d*1), b = (d*1), p = (K*1), s = W*x+b, p = softmax(s)
% cost function: cross entropy loss function + L2-regularization term
% We start by writing a function that loads data into batches:
d = 32*32*3;
K = 10;
lambda = 0;
fileList = {'data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat'};
function [X, Y, y] = LoadBatch(filename)
  inputFile = load(filename); %we load the file
  X = im2double(inputFile.data)'; %octave function that converts an image to doubles.
  y = double(inputFile.labels+1); %we index our classes by 1-10 instead of 0-9
  Y = y == 1:max(y); %Y is 0 if it's not the max and 1 if it is.
  Y = Y';
  y = y';
  N = columns(X); %transpose to fix alignments
  return;
endfunction
function [X, Y, y, N] = LoadAll(fileList)
  X = [];
  Y = [];
  y = [];
  for i=1:length(fileList)
    [Xi,Yi,yi] = LoadBatch(fileList{i});
    X = [X Xi];
    Y = [Y Yi];
    y = [y yi];
    N = columns(X);
  endfor
  y = y';
endfunction
%Test if it works:
% [X,Y,y,N] = LoadBatch('data_batch_1.mat');
[X, Y, y, N] = LoadAll(fileList);

disp("is X correct shape: "),disp(size(X) == [d, N]);
disp("is Y correct shape: "),disp(size(Y) == [K,N]);
disp("is y correct shape: "),disp(size(y) == [N,1]);

% Now we want to initialize the models parameters. We directly write a function that can take
% a initialization type as argument, if none is given it initializes as a zero-mean gaussian
% with 0.1 variance. W = (K*d), b = (K*1).

function [W, b] = Initialize(K, d, initType)
  if nargin < 3 % this check if we have given "initType" as an argument
    W = randn(K,d)*sqrt(0.1); %classic gaussian with zero mean and 0.1 variance
    b = randn(K,1)*sqrt(0.1); %classic gaussian with zero mean and 0.1 variance
    return;
  end
  if initType == 'xavier'
    W = randn(K,d)*sqrt(1/d); %initialized as xavier with 1/N as variance
    b = randn(K,1)*sqrt(1/d); %could be 2/(N_in+N_out), but it's debated.
    return;
  end
endfunction

[W, b] = Initialize(K, d);

disp("Is W the correct shape: "),disp(size(W) == [K,d]);
disp("Is B the correct shape: "),disp(size(b) == [K,1]);
disp("xavier variance should be: "),disp(1/d);
disp("gaussian variance should be: "),disp(0.1);
disp("The variance of W is:"), disp(mean(var(W, 0, 2)));

% Now we define a separate softmax-function that includes considerations for
% numerical stability w.r.t potentially large exponents.
% s as input is W*X + b, (K,d)*(d,N) + (K,1)
function p = Softmax(s)
  s -= max(s);
  p = exp(s) ./ sum(exp(s));
  return;
endfunction

p = Softmax(W*X+b); %broadcasted rendition
disp("Softmax outputs correct shape: "),disp(size(p) == [K,N]);

% We define an evaluation function that utilizes broadcasting
% That is, b takes on the shape of (K,N) for the sake for our affine
% transformation.
function P = EvaluateClassifier(X, W, b)
  s = W*X+b;
  P = Softmax(s);
  return;
endfunction

P = EvaluateClassifier(X, W, b);
disp("EvaluateClassifier can run on 40000 examples: "), disp(size(P));

% We write the cost function as given in the assignment, with cross entropy loss
% and l2 regularization. Y'*P.. comes from that we have a one-hot representation
% of our examples. The products is zero for all 0-elements but 1*probOfClass for the
% correct guess. We therefore optimize by encouraging the correct class to have high prob.

function J = ComputeCost(X, Y, W, b, lambda)
  P = EvaluateClassifier(X, W, b);
  J = -sum(log(sum(Y.*P, 1)))/columns(X) + lambda*sum(sumsq(W)); %changed due to memory issues
  return;
endfunction
J = ComputeCost(X, Y, W, b, lambda);
% We write the accuracy function as one that calculates all our probabilities
% then pick out the highest prob from each example and count all that is correct labels
% we then divide by the size of our dataset and return the percentage as a fraction.
function acc = ComputeAccuracy(X, y, W, b)
  p = EvaluateClassifier(X, W, b);
  [pmax, pmaxidx] = max(p);
  correctLabels = sum(y == pmaxidx');
  acc = correctLabels/columns(X);
  return;
endfunction

acc = ComputeAccuracy(X, y, W, b);
disp("check that acc is in bounds: "),disp(0 <= acc & acc <= 1);


% Now we need to compute our gradients according to the backprop algorithm
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
  % Below is evaluated correct
  % Need to include the derivations for each term, for clarity.
  grad_W = zeros(rows(W), columns(W));
  grad_b = zeros(rows(W), 1);
  for i=1:columns(X)
    x = X(:,i);
    y = Y(:,i);
    p = P(:,i);
    g = -y'/(y'*p)*(diag(p)-p*p'); % We missed a f****** minus sign.
    grad_b = grad_b + g';
    grad_W = grad_W + g'*x';
  endfor
  grad_b = grad_b/columns(X);
  grad_W = grad_W/columns(X) + 2*lambda*W;

  return;
endfunction

% Here we are introducing a resizing function to more quickly adapt our gradient checking.
function [X,Y,y,W] = Resize(X,Y,y,W, size = 2000, dimension = 100)
  X = X(1:dimension,1:size);
  Y = Y(:,1:size);
  y = y(1:size,:);
  W = W(:,1:dimension);
  return
endfunction

function maxDiff = GradChecker(grad_analytic, grad_numeric, epsilon = 1e-6)
  if (size(grad_analytic) != size(grad_numeric))
    maxDiff = 0;
    disp("Please input gradients with corresponding dimensions");
    return;
  endif
  maxDiff = max(max(abs(grad_analytic .- grad_numeric)./max(1e-9, abs(grad_analytic).+abs(grad_numeric))));
  if maxDiff < epsilon
    maxDiff = 1;
  else
    maxDiff = 0;
  endif
  return;
endfunction
% first we resize the vectors
% [X, Y, y,W] = Resize(X,Y,y, W);
% we evaluate again to ensure P is of right size
% P = EvaluateClassifier(X, W, b);
% We compute our tired try of gradients
% lambda = 0;
% [grad_W_analytic, grad_b_analytic] = ComputeGradients(X, Y, P, W, lambda);
% We compute the fast numerical approximation (might wanna test the slow one!)
% [grad_b_numerical, grad_W_numerical] = ComputeGradsNum(X, Y, W, b, lambda, 1e-6);
% we display the differences with a _correct_ gradient function and element-wise operations everywhere,
% as is correct.
% disp("Check that maximum gradient difference is OK for W: "),disp(GradChecker(grad_W_analytic, grad_W_numerical));
% disp("Check that maximum gradient difference is OK for b: "),disp(GradChecker(grad_b_analytic, grad_b_numerical));

function [Wstar, bstar] = MiniBatchGD(X, Y, y,GDparams, W, b, lambda)
  %[n_batch, eta, n_epochs] = GDparams;
  n_batch = GDparams.n_batch;
  eta = GDparams.eta;
  n_epochs = GDparams.n_epochs;
  N = columns(X);
  for i=1:n_epochs
    for j=1:N/n_batch
      j_start = (j-1)*n_batch + 1;
      j_end = j*n_batch;
      inds = j_start:j_end;
      Xbatch = X(:,inds);
      Ybatch = Y(:,inds);
      P = EvaluateClassifier(Xbatch, W, b);
      [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
      W = W-eta*grad_W;
      b = b-eta*grad_b;
    endfor
    disp(ComputeCost(X, Y, W, b, lambda));
    disp(ComputeAccuracy(X, y, W, b));
  endfor
  Wstar = W;
  bstar = b;
  return;
endfunction

GDparams.n_batch = 100;
GDparams.eta = .01;
GDparams.n_epochs = 40;
[W,b] = Initialize(K, d);

[Wstar, bstar] = MiniBatchGD(X,Y,y,GDparams, W, b, 0);
disp(ComputeAccuracy(X,y,Wstar,bstar));
for i=1:K
  im = reshape(Wstar(i, :), 32, 32, 3);
  s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
  s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'size', [1,K]);
pause(50);
