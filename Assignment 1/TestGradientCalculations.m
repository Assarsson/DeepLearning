## This file test the ComputeGradients function defined in the lab spec.
## The lab specification states that ComputeGradients should take in
## X, Y, P, W and lambda as parameters and return [grad_W, grad_b] with dimensions
## [(K, d), (K, 1)]. We have added N as a parameter instead of writing columns(X)
## or size(X,2) inside every function that uses N as a parameter in its calculations.

function correct = TestGradientCalculations()
  % TestGradientCalculations generates new data
  % And ensures that our gradients are relatively close to the numerical
  % approximation. 
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
  [db2, dW2] = ComputeGradsNumSlow(X, Y, W, b, N, lambda, 1e-9);
  maxDiffW = GradChecker(dW1, dW2);
  maxDiffb = GradChecker(db1, db2);
  if maxDiffb > tolerance
    disp('largest gradient difference in W: '),disp(maxDiffW);
    disp('largest gradient difference in b: '),disp(maxDiffb);
    correct = 'fail';
  else
    correct = 'pass';
    disp('largest gradient difference in W: '),disp(maxDiffW);
    disp('largest gradient difference in b: '),disp(maxDiffb);
  endif
  disp(correct);

return;
endfunction
