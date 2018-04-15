function P = EvaluateClassifier(X, Theta)
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
  %   W -- The current weight matrix of size (K, d)
  %   b -- The current bias vector of size (K, 1)
  %
  % OUTPUT:
  %   P -- the probability matrix for the classes of X of size (K, N)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  W1 = Theta{1, 1};
  b1 = Theta{1, 2};
  W2 = Theta{2, 1};
  b2 = Theta{2, 2};
  s1 = W1*X + b1;
  h = Relu(s1);
  s = W2*h + b2;
  P = Softmax(s);
endfunction
