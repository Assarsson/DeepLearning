function correct = TestComputeAccuracy()
  warning('off', 'all');
  disp('Running test for ComputeAccuracy.');
  [X, Y, y, N] = LoadBatch('data_batch_1.mat');
  K = max(y);
  d = rows(X);
  lambda = 0;
  [W,b] = Initialize(K,d);
  acc = ComputeAccuracy(X, y, W, b, N);
  if (size(acc) == [1,1])
    correct = 'pass';
  else
    correct = 'fail';
    disp(size(acc));
  endif
  disp(correct);

endfunction
