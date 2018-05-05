function [grad_b,grad_W] = ComputeGradients(X, Y, cache, mus, vs, W, b, N, lambda)
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
  layers = length(W);
  grad_W = cell(layers,1);
  grad_b = cell(layers,1);
  grad_W = cellfun(@(x, y) double(zeros(size(y))), grad_W, W, 'UniformOutput', false);
  grad_b = cellfun(@(x, y) double(zeros(size(y))), grad_b, b, 'UniformOutput', false);
  P = cache{layers*2, 1};
  for i=1:N
    Xi = X(:,i);
    Yi = Y(:,i);
    Pi = P(:,i);
    g = -(Yi-Pi)'; % We missed a f****** minus sign.
    hi = cache{layers*2-2,1}(:,1);
    s = cache{layers*2-3,1};
    si = s(:,i);
    grad_b{layers,1} += g';
    grad_W{layers, 1} +=g'*hi';
    g = g*W{layers, 1};
    g = g*diag(si > 0);
    for layer = layers-1:-1:2
      hi = cache{layer*2-2,1}(:,i);
      g = BatchNormBackPass(g, s,si, mus{layer}, vs{layer}, N);
      s = cache{layer*2-3,1};
      si = s(:,i);
      grad_b{layer, 1} += g';
      grad_W{layer, 1} += g'*hi'; %+ 2*lambda*W{layer,1};
      g = g*W{layer, 1};
      g = g*diag(si > 0);
    endfor
    g = BatchNormBackPass(g, s, si, mus{1}, vs{1}, N);
    grad_b{1, 1} += g';
    grad_W{1, 1} += g'*Xi'; %+ 2*lambda*W{1,1};
  end
  grad_W = cellfun(@(x) x/N, grad_W, 'UniformOutput', false);
  grad_b = cellfun(@(x) x/N, grad_b, 'UniformOutput', false);
  grad_W = cellfun(@(x, y) x + 2*lambda*y, grad_W, W, 'UniformOutput', false);
endfunction
