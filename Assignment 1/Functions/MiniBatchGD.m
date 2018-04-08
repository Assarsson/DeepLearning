function [Wstar, bstar] = MiniBatchGD(Xbatch, Ybatch,eta, W, b, N,lambda)
  % MiniBatchGD performs the entire forward and backward pass.
  % It divides the dataset (X, Y)-pairs into batches and computes
  % the forward pass to generate a batch-determined class probability distribution
  % P. It then computes the backwardpass gradients and updates the W and b
  % elements, scaled by our learning-rate eta. As W and b isn't dependent on
  % the number of training-examples (N) (check their shapes), they are always
  % full.
  % INPUT:
  %   X -- The current data batch of size (d, N)
  %   Y -- The current one-hot label matrix of size (K, N)
  %   y -- The current label matrix of size (1, N)
  %   GDparams -- Object containing scalars for batchsize, eta and epochs.
  %   W -- The current weight matrix of size (K, d)
  %   b -- The current bias vector of size (K, 1)
  %   N -- The scalar value representing number of examples.
  %   lambda -- The weighting parameter of our regularization term.
  %
  % OUTPUT:
  %   Wstar -- The final computed gradient for W of size (K, d)
  %   bstar -- The final computed gradient for b of size (K, 1)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  P = EvaluateClassifier(Xbatch, W, b);
  [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, N,lambda);
  Wstar = W - eta*grad_W;
  bstar = b - eta*grad_b;
endfunction
