function J = ComputeCost(X, Y, W, b, N, lambda)
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
  [cache, mus, vs] = EvaluateClassifier(X, W, b);
  P = cache{layers*2, 1};
  weightCost = cellfun(@(x) sum(sumsq(x)), W, 'UniformOutput', true);
  regularization_cost = lambda*sum(weightCost);
  J = -sum(log(sum(Y.*P)))/N + regularization_cost;
endfunction
