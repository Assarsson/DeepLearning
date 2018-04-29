function [Xstar, mean_of_X] = Preprocess(X)
  % Preprocess normalizes the dataset by subtracting the mean.
  % This will give the resulting data a zero mean across the examples.
  % INPUT:
  %   X -- An un-normalized data matrix of size (d,N).
  %
  % OUTPUT:
  %   Xstar -- a normalized data matrix of size (d,N)
  %   mean_of_x -- a mean-vector of size (d)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  mean_of_X = mean(X, 2);
  Xstar = X - repmat(mean_of_X, [1, size(X, 2)]);
endfunction
