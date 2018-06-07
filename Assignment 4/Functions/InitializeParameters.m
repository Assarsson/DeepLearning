function RNN = InitializeParameters(K, m)
  sigma = 0.01;
  RNN.b = zeros(m, 1);
  RNN.c = zeros(K, 1);
  RNN.U = randn(m, K)*sigma;
  RNN.W = randn(m, m)*sigma;
  RNN.V = randn(K, m)*sigma;
endfunction
