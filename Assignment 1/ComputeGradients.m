function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, N, lambda)
  % Below is evaluated correct
  % Need to include the derivations for each term, for clarity.
  grad_W = zeros(size(W));
  grad_b = zeros(rows(W), 1);
  for i=1:N
    Xi = X(:,i);
    Yi = Y(:,i);
    Pi = P(:,i);
    g = -Yi'*(diag(Pi)-Pi*Pi')/(Yi'*Pi); % We missed a f****** minus sign.
    grad_b = grad_b + g';
    grad_W = grad_W + g'*Xi';
  end

  grad_b = grad_b/N;
  grad_W = grad_W/N + 2*lambda*W;

  return;
endfunction
