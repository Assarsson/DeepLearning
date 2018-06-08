

%!test
%! addpath Functions/
%! fileName = 'goblet_book.txt'
%! bookData = LoadBatch(fileName);
%! [bookChars, cToIx, ixToC, K] = Preprocess(bookData);
%! hp = GenerateHyperParameters();
%! hp.K = K;
%! [RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx);
%! [P, H, J] = ForwardPass(RNN, X, Y, h0, hp);
%! assert(size(J) == [1,1]);
%! assert(sum(sum(P)) == sum(double(ones(1, hp.seqLength))));
