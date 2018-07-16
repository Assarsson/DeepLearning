

%!test
%! addpath Functions/
%! K = 10;
%! hp = GenerateHyperParameters();
%! bookData = LoadBatch('goblet_book.txt');
%! [bookChars, cToIx, ixToC, K] = Preprocess(bookData);
%! hp.K = K;
%! [RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx, ixToC);
%! assert(size(RNN.b) == [hp.m,1]);
%! assert(size(RNN.c) == [K,1]);
%! assert(size(RNN.U) == [hp.m,K]);
%! assert(size(RNN.W) == [hp.m,hp.m]);
%! assert(size(RNN.V) == [K,hp.m]);
%! assert(RNN.K == K);
%! assert(RNN.m == hp.m);
%! assert(size(x0) == [K, 1]);
%! assert(size(h0) == [hp.m, 1]);
%! assert(size(X) == [K, length(bookData)]);
%! assert(size(Y) == [K, length(bookData)]);
