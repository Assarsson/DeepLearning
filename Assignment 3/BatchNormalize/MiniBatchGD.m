function [Wstar, bstar, Wm, bm] = MiniBatchGD(Xbatch, Ybatch,eta, W, b, Wm, bm, N,lambda, rho)
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
  %   eta -- the current learning rate
  %   W -- The current weight cell of size (2, 1) containing W1 and W2
  %   b -- The current bias cell of size (2, 1) containing b1 and b2
  %   Wm -- The current weight momentum cell of size (2, 1) containing Wm1 and Wm2
  %   bm -- The current bias momentum cell of size (2, 1) containing bm1 and bm2
  %   N -- The scalar value representing number of examples.
  %   lambda -- The weighting parameter of our regularization term.
  %   rho -- The momentum term, a scalar betwenn 0 and 1
  %
  % OUTPUT:
  %   Wstar -- The final computed gradient for W of size (K, d)
  %   bstar -- The final computed gradient for b of size (K, 1)
  %   Wm -- The updated weight momentum cell of size (2, 1) containing Wm1 and Wm2
  %   bm -- The updated bias momentum cell of size (2, 1) containing bm1 and bm2
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  layers = length(W);
  Wstar = cell(layers,1);
  bstar = cell(layers,1);
  [P, S, Shat, H, mus, vs] = EvaluateClassifier(Xbatch, W, b);
  [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch, P, S, Shat, H, mus, vs, W, b, N, lambda);
  Wm = cellfun(@(x, y) rho*x + eta*y, Wm, grad_W, 'UniformOutput', false);
  Wstar = cellfun(@(x, y) x-y, W, Wm, 'UniformOutput', false);
  bm = cellfun(@(x, y) rho*x + eta*y, bm, grad_b, 'UniformOutput', false);
  bstar = cellfun(@(x, y) x-y, b, bm, 'UniformOutput', false);
endfunction
