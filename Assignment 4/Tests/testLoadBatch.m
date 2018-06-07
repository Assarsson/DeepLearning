%!test
%! addpath Functions/
%! fileName = 'goblet_book.txt';
%! bookData = LoadBatch(fileName);
%! assert(size(bookData) == [1, 1107545]);
