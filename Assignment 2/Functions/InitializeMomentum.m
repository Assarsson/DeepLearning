function [Wm, bm] = InitializeMomentum(W, b)
  % Initialize creates our Wm and bm cells and populates them with zeros.
  % INPUT:
  %   W -- the current weight cell of size (2,1) containing W1 and W2
  %   b -- the current bias cell of size (2,1) containing b1 and b2
  %
  % OUTPUT:
  %   Wm -- A populated weight momentum cell of size (2, 1)
  %   bm -- A populated bias momentum cell of size (2, 1)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  Wm = cell(2,1);
  bm = cell(2,1);
  Wm(1,1) = zeros(size(W{1,1}));
  Wm(2,1) = zeros(size(W{2,1}));
  bm(1,1) = zeros(size(b{1,1}));
  bm(2,1) = zeros(size(b{2,1}));

endfunction
