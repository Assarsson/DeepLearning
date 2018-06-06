function J = ComputeCost(X, Y, W, b, N, lambda, varargin)
  % ComputeCost computes the total cost of generating our guessed distribution
  % from the correct distribution of labels, with a cross-entropy loss and
  % and a L2-regularization-term. As the Crossentropy between P and y can be
  % expressed in terms of the Kullback-Leibler-divergence ('KL') between them
  % it can be viewed as calculating the analogous "distance" between
  % the two distributions. (Albeit it is not a true Lp-distance.)
  % Probabilistically, we can view it as that the minimization of that "distance"
  % is the same as maximizing the probability of the data, given our parameters.
  % This is the same as a MLE-estimate, or a MAP-estimate if we assume that the
  % regularization term acts as a Gaussian prior on our parameter space.
  % INPUT:
  %   X -- The current data batch of size (d, N_batch)
  %   Y -- The current one-hot label representation of size (K, N_batch)
  %   W -- The current weight cell of size (2,1) containing W1 and W2
  %   b -- The current bias cell of size (2,1) containing b1 and b2
  %   N -- The current batch size, i.e. N == N_batch
  %   lambda -- The current scalar regularization parameter
  %
  % OUTPUT:
  %   J -- The scalar Cost-value.
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  J = 0;
  layers = length(W);
  if (length(varargin) == 0)
    [P, S, Shat, H, mus, vs] = EvaluateClassifier(X, W, b);
  else
    [P, S, Shat, H, mus, vs] = EvaluateClassifier(X, W, b, varargin{1,1}, varargin{1,2});
  endif
  weightCost = cellfun(@(x) sum(sumsq(x)), W, 'UniformOutput', true);
  regCost = lambda*sum(weightCost);
  J = -sum(log(sum(Y.*P)))/N + regCost;
endfunction
