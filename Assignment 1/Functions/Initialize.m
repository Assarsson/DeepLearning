function [W, b] = Initialize(K, d, initType)
  % Initialize creates our W and b matrices and populates them with random values.
  % It either utilizes a random gaussian prior on the parameters or a Xavier prior.
  % The role of the Xavier prior is to keep X and and W*X equivariant, to increase
  % stability of the model and increase speed in our iterations.
  % INPUT:
  %   K -- The number of classes for our problem
  %   d -- The dimensionality of our input X.
  %   initType -- Optional variable that can invoke xavier-initialization.
  %
  % OUTPUT:
  %   W -- A populated weight matrix of size (K, d)
  %   b -- A populated bias vector of size (K, 1)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  W = double(randn(K,d));
  b = double(randn(K,1));
  mu = 0;
  if nargin < 3
    variance = 0.01;
  elseif (initType == 'xavier')
    variance = 1/d;
  elseif (initType == 'norand')
    variance = 0.1;
    W = double(ones(K,d));
    b = double(ones(K,1));
  end
  W = W*sqrt(variance) + mu;
  b = b*sqrt(variance) + mu;
  return;
endfunction
