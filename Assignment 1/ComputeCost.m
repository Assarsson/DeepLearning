function J = ComputeCost(X, Y, W, b, N, lambda)
  P = EvaluateClassifier(X, W, b);
  J = -sum(diag(log(Y'*P)))/N + lambda*sum(sumsq(W));
  return;
endfunction
