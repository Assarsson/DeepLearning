function [RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx, ixToC, testing)
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
  if (testing == true)
    RNN.N = 25;
  endif
  x0 = zeros(K, 1);
  h0 = zeros(hp.m, 1);
  X = zeros(RNN.K, RNN.N);
  Y = zeros(RNN.K, RNN.N);

  %convert to one-hot encoding
  Xchars = bookData(1:hp.seqLength);
  Ychars = bookData(2:hp.seqLength+1);
  for i = 1:RNN.N
    X(cToIx.(bookData(i)), i) = 1;
    Y(cToIx.(bookData(i)), i) = 1;
  endfor
endfunction
