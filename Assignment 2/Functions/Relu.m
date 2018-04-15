function h = Relu(s)
  % This activation function is the relu-activation function.
  % INPUT:
  %   s -- The result of an affine W*X+b-transformation of size (m, N)
  %
  % OUTPUT:
  %   h --  non-linearity output of size (m, N)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  h = max(0, s);
endfunction
