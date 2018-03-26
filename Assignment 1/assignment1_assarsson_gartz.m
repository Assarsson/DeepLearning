## Information: This program follows the instruction for lab 1 in the Masters Course
## of Deep Learning at the Royal Institute of Technology. It is a multi-class one-layer
## neural network trained with mini-batch gradient descent on both a cross-entropy-loss
## against a softmax output, as well as a SVM multi-class loss.
## Author: Fabian Assarsson & Madeleine Gartz
## Date: 2018-03-25

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

[W, b] = Initialize(K, d, 'xavier');

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

P = EvaluateClassifier(X(:,1:100), W, b);
disp("EvaluateClassifier can run on 100 examples: "), disp(size(P)(2) == 100 );

% We write the cost function as given in the assignment, with cross entropy loss
% and l2 regularization. Y'*Eval.. comes from that we have a one-hot representation
% of our examples.

function J = ComputeCost(X, Y, W, b, lambda = 0.01)
  J = 1/columns(X)*sum(-log(Y'*EvaluateClassifier(X, W, b))) + lambda*sum(sum(W.**2));
  return;
endfunction
