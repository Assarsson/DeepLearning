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