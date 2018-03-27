

function [X, Y, y, N] = LoadBatch(fileName)
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
