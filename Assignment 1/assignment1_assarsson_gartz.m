## Information: This program follows the instruction for lab 1 in the Masters Course
## of Deep Learning at the Royal Institute of Technology. It is a multi-class one-layer
## neural network trained with mini-batch gradient descent on both a cross-entropy-loss
## against a softmax output, as well as a SVM multi-class loss.
## Author: Fabian Assarsson & Madeleine Gartz
## Date: 2018-03-25

rand('state',1);
% First we start with looking at some of the images from the dataset:
addpath Datasets/cifar-10-batches-mat/; % adds our path to data
A = load('data_batch_1.mat'); % loads a saved datafile with examples as rows in A
I = reshape(A.data', 32, 32, 3, 10000); % We reshape it into a tensor for the montage-function
I = permute(I, [2, 1, 3, 4]); % We permute the first and second axis
montage(I(:,:,:,1:500), 'Size', [5,5]); % We display it with our helper function

% Exercise 1
% Given: W = (K*d), x = (d*1), b = (d*1), p = (K*1), s = W*x+b, p = softmax(s)
% cost function: cross entropy loss function + L2-regularization term
% We start by writing a function that loads data into batches:
d = 32*32*3;
N = 10000;
K = 10;
lambda = 0.01;
function [X, Y, y] = LoadBatch(filename)
  inputFile = load(filename); %we load the file
  X = im2double(inputFile.data'); %octave function that converts an image to doubles.
  y = inputFile.labels+1; %we index our classes by 1-10 instead of 0-9
  Y = y == 1:max(y); %Y is 0 if it's not the max and 1 if it is.
  Y = Y'; %transpose to fix alignments
  return;
endfunction

%Test if it works:
[X,Y,y] = LoadBatch('data_batch_1.mat');

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
disp("EvaluateClassifier can run on 10000 examples: "), disp(size(P));

% We write the cost function as given in the assignment, with cross entropy loss
% and l2 regularization. Y'*P.. comes from that we have a one-hot representation
% of our examples. The products is zero for all 0-elements but 1*probOfClass for the
% correct guess. We therefore optimize by encouraging the correct class to have high prob.

function J = ComputeCost(X, Y, W, b, lambda)
  P = EvaluateClassifier(X, W, b);
  J = (1/columns(X))*sum(-log(diag(Y'*P))) + lambda*sum(sum(W.**2));
  return;
endfunction
J = ComputeCost(X, Y, W, b, lambda);
% We write the accuracy function as one that calculates all our probabilities
% then pick out the highest prob from each example and count all that is correct labels
% we then divide by the size of our dataset and return the percentage as a fraction.
function acc = ComputeAccuracy(X, y, W, b)
  p = EvaluateClassifier(X, W, b);
  [pmax, pmaxidx] = max(p);
  correctLabels = sum(pmaxidx' == y);
  acc = correctLabels/columns(X);
  return;
endfunction

acc = ComputeAccuracy(X, y, W, b);
disp("check that acc is in bounds: "),disp(0 <= acc & acc <= 1);


% Now we need to compute our gradients according to the backprop algorithm
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
  % Below needs to be checked, with an additional derivation. A tired attempt :P!
  grad_W = zeros(rows(W), columns(W));
  grad_b = zeros(rows(W), 1);
  for i=1:columns(X)
    x = X(:,i);
    y = Y(:,i);
    p = P(:,i);
    g = y'/(y'*p)*(diag(p)-p*p');
    grad_b = grad_b + g';
    grad_W = grad_W + g'*x';
  endfor
  grad_b = grad_b/columns(X);
  grad_W = grad_W/columns(X) + 2*lambda*W;

  return;
endfunction

% Here we are introducing a resizing function to more quickly adapt our gradient checking.
function [X,Y,y,W] = Resize(X,Y,y,W, size = 32, dimension = 1000)
  X = X(1:dimension,1:size);
  Y = Y(:,1:size);
  y = y(1:size,:);
  W = W(:,1:dimension);
  return
endfunction

% first we resize the vectors
[X, Y, y,W] = Resize(X,Y,y, W);
% we evaluate again to ensure P is of right size
P = EvaluateClassifier(X, W, b);
% We compute our tired try of gradients
lambda = 0;
[grad_W_analytic, grad_b_analytic] = ComputeGradients(X, Y, P, W, lambda);
% We compute the fast numerical approximation (might wanna test the slow one!)
[grad_b_numerical, grad_W_numerical] = ComputeGradsNum(X, Y, W, b, lambda, 1e-6);
% we display the differences. The diagonal is one, and I have no idea any more. Good night.
disp(abs(grad_W_analytic .- grad_W_numerical)/max(1e-9, abs(grad_W_analytic) .+ abs(grad_W_analytic)));
