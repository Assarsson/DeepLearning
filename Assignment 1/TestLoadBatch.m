## This file test the LoadBatch defined in the lab spec.
## The lab specification states that LoadBatch should take in
## a filename as a parameter and return [X, Y, y] with dimensionalities
## [(d, N), (K, N), (1, N)].

function correct = TestLoadBatch()
  % TestLoadBatch loads fresh data and ensures that LoadBatch
  % returns the correct dimensions in alla variables.
  disp('Running load-batch-test with all ones indicating full pass.');
  [X, Y, y, N] = LoadBatch('data_batch_1.mat');
  disp(size(X) == [3072,N]);
  disp(size(Y) == [10,N]);
  disp(size(y) == [1,N]);
  correct = 'pass';
endfunction
