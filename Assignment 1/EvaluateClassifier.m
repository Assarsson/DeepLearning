function P = EvaluateClassifier(X, W, b)
  s = W*X + b;
  P = Softmax(s);
  return;
endfunction
