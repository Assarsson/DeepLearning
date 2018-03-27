function correct = TestEvaluateClassifier()
  disp('Running test for EvaluateClassifier.')
  [X, Y, y, N] = LoadBatch('data_batch_1.mat');
  K = max(y);
  d = rows(X);
  [W,b] = Initialize(K,d);
  P = EvaluateClassifier(X, W, b);
  s = W*X+b;
  s -= max(s);
  s = exp(s);
  Pi = bsxfun(@rdivide, s, sum(s));
  if size(P) == [K, N] && norm(P.-Pi) == 0
    correct = 'pass';
  else
    correct = 'fail';
    disp(size(P));
  end
  disp(correct);


  return;

endfunction
