function [RNN, x0, h0, X, Y] = InitializeParameters(K, hp)
  sigma = 0.01;
  RNN.b = zeros(hp.m, 1);
  RNN.c = zeros(K, 1);
  RNN.U = randn(hp.m, K)*sigma;
  RNN.W = randn(hp.m, hp.m)*sigma;
  RNN.V = randn(K, hp.m)*sigma;
  RNN.K = K;
  RNN.m = hp.m;
  x0 = zeros(K, 1);
  h0 = zeros(hp.m, 1);
  X = zeros(K, hp.seqLength);
  Y = zeros(K, hp.seqLength);
endfunction
