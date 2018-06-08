function J = ComputeCost(Y, P)
  J = -sum(log(sum(Y.*P, 1)));
endfunction
