function [X,Y,y,W,N] = Resize(X,Y,y,W, size = 100, dimension = 3072)
  % Resize can reshape the dataset on the dimension and batch-size axis.
  % It is an auxilliary function to be used for quicker testing.
  % INPUT:
  %   X -- The current data batch of size (d, N_batch)
  %   Y -- The current one-hot label matrix of size (K, N)
  %   y -- The current ground truth label matrix of size (1, N)
  %   W -- The current weight matrix of size (K, d)
  %   size -- The number of examples to resize down to.
  %   dimension -- the required output dimension of this function.
  %
  % OUTPUT:
  %   X -- X but resized
  %   Y -- Y but resized
  %   y -- y but resized
  %   W -- W but resized
  %   N -- scalar to reflect reduced number of examples.
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  X = X(1:dimension,1:size);
  Y = Y(:,1:size);
  y = y(:,1:size);
  W = W(:,1:dimension);
  N = size;
  return
endfunction
