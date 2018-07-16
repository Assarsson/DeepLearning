function [RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx, ixToC)
  sigma = 0.01;
  RNN.b = zeros(hp.m, 1);
  RNN.c = zeros(K, 1);
  RNN.U = randn(hp.m, K)*sigma;
  RNN.W = randn(hp.m, hp.m)*sigma;
  RNN.V = randn(K, hp.m)*sigma;
  RNN.K = K;
  RNN.m = hp.m;
  RNN.cToIx = cToIx;
  RNN.ixToC = ixToC;
  [~, RNN.N] = size(bookData);
  x0 = zeros(K, 1);
  h0 = zeros(hp.m, 1);
  X = zeros(K, hp.seqLength);
  Y = zeros(K, hp.seqLength);

  %convert to one-hot encoding
  Xchars = bookData(1:hp.seqLength);
  Ychars = bookData(2:hp.seqLength+1);
  for i = 1:hp.seqLength
    X(cToIx.(Xchars(i)), i) = 1;
    Y(cToIx.(Ychars(i)), i) = 1;
  endfor
endfunction
