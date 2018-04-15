function [Theta] = Initialize(K, d, hiddenNodes, initType)
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
  Theta = cell(2,2);
  W1 = double(randn(hiddenNodes,d));
  W2 = double(randn(K, hiddenNodes));
  b1 = double(zeros(hiddenNodes,1));
  b2 = double(zeros(K,1));
  mu = 0;
  variance = 0.000001;
  if nargin < 4
    variance = 0.000001;
  elseif (initType == 'xavier')
    variance = 1/d;
  elseif (initType == 'norand')
    variance = 0.1;
    W = double(ones(K,d));
    b = double(ones(K,1));
  end
  W1 = W1*sqrt(variance) + mu;
  W2 = W2*sqrt(variance) + mu;
  Theta(1,1) = W1;
  Theta(1,2) = b1;
  Theta(2,1) = W2;
  Theta(2,2) = b2;
  
endfunction
