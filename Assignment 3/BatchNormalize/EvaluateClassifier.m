function [P, S, Shat, H, mus, vs] = EvaluateClassifier(X, W, b, varargin)
  % EvaluateClassifier computes the two layers of our forward pass. It is described
  % classicaly by an affine transformation that consists of a linear map and bias term.
  % W*X "twists and turns" the data, whereas the bias term translates it.
  % We then add our preferedd non-linearity, this time defined by the Relu model.
  % The relu model has non-saturating gradients and helps with disentanglement, representation
  % linear separability and "bagging". See 'Deep Sparse Rectifier Neural Networks' for more
  % information.
  % As our job is to find 1 of K classes that our image belongs to, our output
  % needs to reflect, in someway, or confidence in the different classes.
  % The softmax function produces a probability distribution over K classes
  % in the sense that each value is confined in [0,1] and sum(P) == 1. As it contains
  % exponentiations and normalizations, we conclude that our affine transformation
  % produces un-normalized (normalized by the division) log-probabilities (converted with exp).
  % This also explains the choice of cross-entropy loss, as that loss function minimizes
  % distributional distances between our guessed distribution from this function, and our
  % true labels in Y or y.
  % INPUT:
  %   X -- The current data batch of size (d, N_batch)
  %   W -- The current weight cell of size (2, 1) containing W1 and W2
  %   b -- The current bias cell of size (2, 1) containing b1 and b2
  %
  % OUTPUT:
  %   P -- the probability matrix for the classes of X of size (K, N)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  isTrain = length(varargin) == 0;
  layers = length(W);
  parameters = layers*2;
  S = cell(layers, 1);
  Shat = cell(layers-1, 1);
  H = cell(layers-1, 1);
  if (isTrain)
    mus = cell(layers, 1);
    vs = cell(layers, 1);
  else
    mus = varargin{1,1};
    vs = varargin{1,2};
  endif

  for layer = 1:(layers-1)
    s = W{layer, 1}*X + b{layer, 1};
    S(layer, 1) = s;
    if (isTrain)
      mu = ComputeMean(s);
      mus(layer, 1) = mu;
      v = ComputeVariance(s,mu);
      vs(layer, 1) = v;
    endif
    s = BatchNormalize(s, mus{layer,1}, vs{layer,1});
    Shat{layer, 1} = s;
    h = Relu(s);
    H(layer, 1) = h;
    X = h;
  endfor
  s = W{layers, 1}*X + b{layers, 1};
  S{layers, 1} = s;
  P = Softmax(s);
endfunction
