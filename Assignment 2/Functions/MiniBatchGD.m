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
  Wstar = cell(2,1);
  bstar = cell(2,1);
  cache = EvaluateClassifier(Xbatch, W, b);
  [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch, cache, W,b, N,lambda);
  Wm(1,1) = rho*Wm{1,1} + eta*grad_W{1,1};
  Wstar(1,1) = W{1,1}-Wm{1,1};
  %Wstar(1,1) = W{1,1}-eta*grad_W{1,1};

  Wm(2,1) = rho*Wm{2,1} + eta*grad_W{2,1};
  Wstar(2,1) = W{2,1} - Wm{2,1};
  %Wstar(2,1) = W{2,1}-eta*grad_W{2,1};

  bm(1,1) = rho*bm{1,1} + eta*grad_b{1,1};
  bstar(1,1) = b{1,1}-bm{1,1};
  %bstar(1,1) = b{1,1}-eta*grad_b{1,1};

  bm(2,1) = rho*bm{2,1} + eta*grad_b{2,1};
  bstar(2,1) = b{2,1} - bm{2,1};
  %bstar(2,1) = b{2,1}-eta*grad_b{2,1};
endfunction
