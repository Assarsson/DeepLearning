function p = Softmax(s)
  % This softmax implementation follow the standard scheme for such a function.
  % It's range is constrained to [0,1] and by summing over the exponents in the
  % denominator, we ensure a 'proper' distribution in a Kolmogorovian sense.
  % The commented line is to increase numerical stability once the gradients are
  % correct.
  % INPUT:
  %   s -- The result of an affine W*X+b-transformation of size (K, N)
  %
  % OUTPUT:
  %   p -- the probability matrix for the classes of X of size (K, N)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  %s -= max(s);
  p = exp(s) ./ sum(exp(s));
endfunction
