function params = Generateparams(min_value, max_value, no_params)
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
params = 10.^(min_value+(max_value-min_value)*rand(no_params,1)');
endfunction
