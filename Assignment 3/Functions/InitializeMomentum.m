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
  layers = length(W);
  Wm = cell(layers,1);
  bm = cell(layers,1);
  for layer = 1:layers
    Wm(layer, 1) = zeros(size(W{layer, 1}));
    bm(layer, 1) = zeros(size(b{layer, 1}));
  endfor
  
endfunction
