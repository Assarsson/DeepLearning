

%!test
%! addpath Functions/
%! fileName = 'goblet_book.txt'
%! bookData = LoadBatch(fileName);
%! [bookChars, cToIx, ixToC, K] = Preprocess(bookData);
%! hp = GenerateHyperParameters();
%! [RNN, x0, h0] = InitializeParameters(K, hp.m);
%! [P, H, Y] = Synthesize(RNN, h0, x0, hp.seqLength);
%! text = OneHotParser(Y, ixToC);
%! disp(text);
