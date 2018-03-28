function [X, Y, y, N] = LoadBatch(fileName)
  % LoadBatch loads a batch of data as defined in the lab instructions.
  % It reads from that loaded file and puts the data in our X variable
  % our labels in the y variable and then additionally creates a one-hot
  % representation of y and saves it to Y.
  % INPUT:
  %   fileName -- a string containing a .mat-file in the path-environment.
  %
  % OUTPUT:
  %   X -- Matrix of data with size (d, N)
  %   Y -- Matrix of one-hot repesentation with size (K, N)
  %   y -- Vector of ground-truth labels of size (1, N)
  %   N -- Scalar value with value N. For readibility in subsequent functions.
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  warning('off','all');
  addpath Datasets/cifar-10-batches-mat/;
  inputFile = load(fileName);
  X = im2double(inputFile.data)';
  %X = double(inputFile.data)'/255;
  y = double(inputFile.labels') +1;
  Y = y' == 1:max(y);
  N = columns(X);
  Y = Y';
  return;
endfunction
