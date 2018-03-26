% Information: This program follows the instruction for lab 1 in the Masters Course
% of Deep Learning at the Royal Institute of Technology. It is a multi-class one-layer
% neural network trained with mini-batch gradient descent on both a cross-entropy-loss
% against a softmax output, as well as a SVM multi-class loss.
% Authors: Fabian Assarsson & Madeleine Gartz
% Date: 2018-03-25

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
