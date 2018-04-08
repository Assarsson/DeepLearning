## This file test the ComputeAccuracy defined in the lab spec.
## The lab specification states that ComputeAccuracy should take in
## X, y, W, b as parameters and return acc as a scalar value. We have added
## N as a parameter instead of writing columns(X) or size(X,2) inside every function
## that uses N as a parameter in its calculations.

function correct = TestComputeAccuracy()
  % TestComputeAccuracy runs a test on newly generated data
  % It ensures that accuracy is a scalar value.
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
