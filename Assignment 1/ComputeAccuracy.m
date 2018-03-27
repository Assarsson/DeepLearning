function acc = ComputeAccuracy(X, y, W, b, N)
  P = EvaluateClassifier(X, W, b);
  [_,PMaxIdx] = max(P);
  acc = sum(y == PMaxIdx);
  acc = acc/N;
  return;
endfunction
