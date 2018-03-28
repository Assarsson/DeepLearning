## This file test the MiniBatchGD function defined in the lab spec.
## The lab specification states that EvaluateClassifier should take in
## X,Y, GDparams, W, b and lambda as parameters and return [Wstar, bstar] with
## dimensionalities [(K, d), (K, 1)]. We have again included the N parameter
## in this function to avoid columns(X) or size(X, 2) in the functions that require
## number of examples as parameter in their calculations.

function correct = TestMiniBatchGD()
  % TestMiniBatchGD generates new data and initializes W and b.
  % It sets GDparams according to the specification as an object
  % and then runs MiniBatchGD and ensures that output is correctly
  % dimensioned.
  disp('Running test on MiniBatchGD-algorithm');
  [X, Y, y, N] = LoadBatch('data_batch_1.mat');
  K = max(y);
  d = rows(X);
  lambda = 0;
  [W, b] = Initialize(K, d);
  GDparams.n_batch = 100;
  GDparams.eta = .01;
  GDparams.n_epochs = 20;
  [Wstar, bstar] = MiniBatchGD(X, Y, y,GDparams, W, b, N, lambda);
  if (size(Wstar) = size(W) && size(bstar) == size(b))
    correct = 'pass';
  else
    correct = 'fail';
  disp(correct);
  end
endfunction
