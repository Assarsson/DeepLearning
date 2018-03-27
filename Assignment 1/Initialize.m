function [W, b] = Initialize(K, d, initType)
  W = randn(K,d);
  b = randn(K,1);
  if nargin < 3
    W = W*sqrt(0.1);
    b = b*sqrt(0.1);
    return;
  end
  if (initType == 'xavier')
    W = W*sqrt(1/d);
    b = b*sqrt(1/d);
    return;
  end
endfunction
