%!test
%! addpath Functions/
%! fileName = 'goblet_book.txt';
%! bookData = LoadBatch(fileName);
%! assert(size(bookData) == [1, 1107545]);
%! [bookChars, cToIx, ixToC] = Preprocess(bookData);
%! assert(size(cToIx) == size(ixToC));
%! assert(cToIx.(ixToC.('60')) == 60);
%! assert(size(bookChars) == [1, 83]);
