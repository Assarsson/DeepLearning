function [Wm, bm] = InitializeMomentum(W, b)
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
  Wm = cell(2,1);
  bm = cell(2,1);
  Wm(1,1) = zeros(size(W{1,1}));
  Wm(2,1) = zeros(size(W{2,1}));
  bm(1,1) = zeros(size(b{1,1}));
  bm(2,1) = zeros(size(b{2,1}));

endfunction
