%!test
%! addpath Functions/
%! fileName = 'goblet_book.txt';
%! bookData = LoadBatch(fileName);
%! hp = GenerateHyperParameters();
%! [bookChars, cToIx, ixToC, K] = Preprocess(bookData);
%! hp.K = K;
%! [RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx);
%! [P, H, J] = ForwardPass(RNN, X, Y, h0, hp);
%! gradients = BackwardPass(RNN, X, Y, P, H, hp);
%! fields = fieldnames(RNN);
%! for i = 1:length(fields)
%!  assert(size(gradients.(fields{i})) == size(RNN.(fields{i})));
%! endfor
