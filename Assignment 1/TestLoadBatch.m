function correct = TestLoadBatch()
  disp('Running load-batch-test with all ones indicating full pass.');
  [X, Y, y, N] = LoadBatch('data_batch_1.mat');
  disp(size(X) == [3072,N]);
  disp(size(Y) == [10,N]);
  disp(size(y) == [1,N]);
  correct = 'pass';
endfunction
