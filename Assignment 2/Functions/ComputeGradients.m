function [grad_b,grad_W] = ComputeGradients(X, Y, cache, W, b, N, lambda)
  % ComputeGradients computes the gradients of W and b as defined by differentiating
  % the cost function with respects to the node in the computational graph and traversing it
  % likeso, until we reach the parameter variable of interest. As we have a 2-layer network
  % this results in us needing to update DL/DW1, DL/DW2, DL/Db1 and DL/Db2. 
  % INPUT:
  %   X -- The current data batch of size (d, N_batch)
  %   Y -- The current one-hot label representation of size (K, N_batch) (true distribution)
  %   P -- The current probability distribution calculated by our parametrized model.
  %   W -- The current weight cell of size (2, 1) containing W1 and W2
  %   b -- The current bias cell of size (2,1) containing b1 and b2
  %   N -- The current batch size, i.e. N == N_batch
  %   lambda -- The current scalar regularization parameter
  %
  % OUTPUT:
  %   grad_W -- The gradient w.r.t W of size (K, d)
  %   grad_b -- The gradient w.r.t b of size (K, 1)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  grad_W = cell(2,1);
  grad_b = cell(2,1);
  grad_W1 = zeros(size(W{1,1}));
  grad_b1 = zeros(size(b{1,1}));
  grad_W2 = zeros(size(W{2,1}));
  grad_b2 = zeros(size(b{2,1}));
  h = cache{2,1};
  W2 = W{2,1};
  W1 = W{1,1};
  s1 = cache{1,1};
  P = cache{4,1};
  for i=1:N
    Xi = X(:,i);
    Yi = Y(:,i);
    Pi = P(:,i);
    hi = h(:,i);
    s1i = s1(:,i);
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi); % We missed a f****** minus sign.
    grad_b2 += g';
    grad_W2 += g'*hi';
    g = g*W2;
    g = g*diag(s1i > 0);
    grad_b1 += g';
    grad_W1 += g'*Xi';
  end

  grad_b1 /= N;
  grad_W1 /= N;
  grad_b2 /= N;
  grad_W2 /= N;
  grad_W1 += 2*lambda*W1;
  grad_W2 += 2*lambda*W2;
  grad_W(1,1) = grad_W1;
  grad_b(1,1) = grad_b1;
  grad_W(2,1) = grad_W2;
  grad_b(2,1) = grad_b2;
endfunction
