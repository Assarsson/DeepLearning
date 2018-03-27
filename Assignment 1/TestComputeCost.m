function correct = TestComputeCost()
  disp('Running test for cost dimensionality.');
  [X, Y, y, N] = LoadBatch('data_batch_1.mat');
  K = max(y);
  d = rows(X);
  lambda = 0;
  [W, b] = Initialize(K, d);
  cost = ComputeCost(X, Y, W, b, N, 0);
  if (size(cost) == [1,1])
    correct = 'pass';
  else
    correct = 'fail';
  endif
  disp(correct);
  return;

endfunction
