function correct = TestMiniBatchGD()
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
