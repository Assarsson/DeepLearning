function correct = TestGradientCalculations()
  disp('Running test for Gradient Calculations.');
  [X, Y, y, N] = LoadBatch('data_batch_1.mat');
  K = max(y);
  d = rows(X);
  lambda = 0;
  tolerance = 1e-4;
  [W, b] = Initialize(K, d);
  [X, Y, y, W, N] = Resize(X, Y, y, W);
  P = EvaluateClassifier(X, W, b);
  [dW1, db1] = ComputeGradients(X, Y, P, W, N, lambda);
  [db2, dW2] = ComputeGradsNum(X, Y, W, b, N, lambda, 1e-9);
  maxDiffW = GradChecker(dW1, dW2);
  maxDiffb = GradChecker(db1, db2);
  if maxDiffb > tolerance
    disp('largest gradient difference in W: '),disp(maxDiffW);
    disp('largest gradient difference in b: '),disp(maxDiffb);
    correct = 'fail';
  else
    correct = 'pass';
  endif
  disp(correct);

return;
endfunction
