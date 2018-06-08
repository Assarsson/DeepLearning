
function [bookChars, cToIx, ixToC, K] = Preprocess(bookData)
  % This function acts as a preprocessing step for the data
  % It takes in the textdata and generates a vector of unique characters,
  % and two maps between the indices and the characters for later use.
  %
  % INPUT:
  %   bookData -- vector containing the full book
  %
  % OUTPUT:
  %   bookChars -- A vector containing the unique characters in the book
  %   xToIx -- a Map that maps from characters to their index
  %   ixToC -- a Map that maps from indices to their character.
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  bookChars = unique(bookData);
  cs = bookChars;
  K = length(bookChars);
  cToIx = struct();
  ixToC = struct();
  for c = cs
    ix = find(cs == c);
    cToIx.(c) = ix;
    ixToC.(num2str(ix)) = c;
  endfor
endfunction
