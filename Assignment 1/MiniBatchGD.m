function [Wstar, bstar] = MiniBatchGD(X, Y, y,GDparams, W, b, N,lambda)
  %[n_batch, eta, n_epochs] = GDparams;
  n_batch = GDparams.n_batch;
  eta = GDparams.eta;
  n_epochs = GDparams.n_epochs;
  for i=1:n_epochs
    for j=1:N/n_batch
      j_start = (j-1)*n_batch + 1;
      j_end = j*n_batch;
      inds = j_start:j_end;
      Xbatch = X(:,inds);
      Ybatch = Y(:,inds);
      P = EvaluateClassifier(Xbatch, W, b);
      [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, n_batch,lambda);
      W = W-eta*grad_W;
      b = b-eta*grad_b;
    endfor
    disp(ComputeCost(X, Y, W, b, N,lambda));
    disp(ComputeAccuracy(X, y, W, b, N));
  endfor
  Wstar = W;
  bstar = b;
  return;
endfunction
