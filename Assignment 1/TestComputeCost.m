## This file test the ComputeCost function defined in the lab spec.
## The lab specification states that ComputeCost should take in
## X, y, W, b and lambda as parameters and return J as a scalar value. We have added
## N as a parameter instead of writing columns(X) or size(X,2) inside every function
## that uses N as a parameter in its calculations.

function correct = TestComputeCost()
  % TestComputeCost runs a test on newly generated data
  % and ensures that cost is a scalar value.
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
