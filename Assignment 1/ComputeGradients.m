function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, N, lambda)
  % ComputeGradients computes the gradients of W and b as defined by differentiating
  % the cost function with respects to the node in the computational graph and traversing it
  % likeso, until we reach the parameter variable of interest. As we have a 1-layer shallow network
  % this results in us only needing to update DL/DW and DL/Db. The gradient of b is defined solely
  % by the gradient-function g(), stemming from the cross-entropy loss function. W has the
  % inner derivative from multiplication with X, as well as the derivative of the reguralization
  % term.
  % INPUT:
  %   X -- The current data batch of size (d, N_batch)
  %   Y -- The current one-hot label representation of size (K, N_batch) (true distribution)
  %   P -- The current probability distribution calculated by our parametrized model.
  %   W -- The current weight matrix of size (K, d)
  %   N -- The current batch size, i.e. N == N_batch
  %   lambda -- The current scalar regularization parameter
  %
  % OUTPUT:
  %   grad_W -- The gradient w.r.t W of size (K, d)
  %   grad_b -- The gradient w.r.t b of size (K, 1)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  grad_W = zeros(size(W));
  grad_b = zeros(rows(W), 1);
  for i=1:N
    Xi = X(:,i);
    Yi = Y(:,i);
    Pi = P(:,i);
    g = -Yi'*(diag(Pi)-Pi*Pi')/(Yi'*Pi); % We missed a f****** minus sign.
    grad_b += g';
    grad_W += g'*Xi';
  end

  regularization_term = 2*lambda*W;
  grad_b /= N;
  grad_W /= N;
  grad_W += regularization_term;

  return;
endfunction
