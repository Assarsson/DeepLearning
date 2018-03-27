function correct = TestEvaluateClassifier()
  disp('Running test for EvaluateClassifier.')
  [X, Y, y, N] = LoadBatch('data_batch_1.mat');
  K = max(y);
  d = rows(X);
  [W,b] = Initialize(K,d);
  P = EvaluateClassifier(X, W, b);
  s = W*X+b;
  s = exp(s);
  Pi = bsxfun(@rdivide, s, sum(s));
  if size(P) == [K, N] && norm(P.-Pi) == 0
    correct = 'pass';
  else
    correct = 'fail';
    disp('size'), disp(size(P));
    disp('norm'), disp(norm(P.-Pi));
  end
  disp(correct);


  return;

endfunction
