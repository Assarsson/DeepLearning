

%!test
%! addpath Functions/
%! K = 10;
%! m = 20;
%! RNN = InitializeParameters(K, m);
%! assert(size(RNN.b) == [m,1]);
%! assert(size(RNN.c) == [K,1]);
%! assert(size(RNN.U) == [m,K]);
%! assert(size(RNN.W) == [m,m]);
%! assert(size(RNN.V) == [K,m]);
