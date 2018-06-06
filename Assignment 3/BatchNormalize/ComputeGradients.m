function [grad_b,grad_W] = ComputeGradients(X, Y, P, S, Shat, H, mus, vs, W, b, N, lambda)
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
  g = cell(N, 1);
  for i=1:N
    Yi = Y(:,i);
    Pi = P(:,i);
    g{i} = -(Yi-Pi)'; % We missed a f****** minus sign.
    hi = H{layers-1,1}(:,i);
    si = Shat{layers-1,1}(:,i);
    grad_b{layers,1} += g{i}';
    grad_W{layers,1} += g{i}'*hi';
    g{i} = g{i}*W{layers, 1};
    g{i} = g{i}*diag(si > 0);
  end
  %grad_W{layers, 1} = grad_W{layers, 1}/N + 2*lambda*W{layers, 1};
  %grad_b{layers, 1} = grad_b{layers,1}/N;
  for layer = layers-1:-1:1
    disp(layer);
    g = BatchNormBackPass(g, S{layer}, mus{layer}, vs{layer}, N);
    if (layer == 1)
      h = X;
    else
      h = H{layer-1,1};
    endif
    for i = 1:N
      grad_b{layer, 1} += g{i}';
      grad_W{layer, 1} += g{i}'*h(:,i)';
      if (layer != 1)%+ 2*lambda*W{layer,1};
        g{i} = g{i}*W{layer, 1};
        si = Shat{layer-1,1}(:,i);
        g{i} = g{i}*diag(si > 0);
      endif
    endfor
  endfor
  grad_b = cellfun(@(x) x/N, grad_b, 'UniformOutput', false);
  grad_W = cellfun(@(x, y) x/N + 2*lambda*y, grad_W, W, 'UniformOutput', false);

endfunction
