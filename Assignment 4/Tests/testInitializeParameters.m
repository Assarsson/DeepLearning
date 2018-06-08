

%!test
%! addpath Functions/
%! K = 10;
%! m = 20;
%! [RNN, x0, h0] = InitializeParameters(K, m);
%! assert(size(RNN.b) == [m,1]);
%! assert(size(RNN.c) == [K,1]);
%! assert(size(RNN.U) == [m,K]);
%! assert(size(RNN.W) == [m,m]);
%! assert(size(RNN.V) == [K,m]);
%! assert(RNN.K == K);
%! assert(RNN.m == m);
%! assert(size(x0) == [K, 1]);
%! assert(size(h0) == [m, 1]);
