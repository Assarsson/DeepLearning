function J = ComputeCost(Y, P)
  % This function computes the cost based on the cross-entropy loss function
  %
  % INPUT:
  %   Y   -- A one-hot-matrix on character level for the output data
  %   P   -- A probability matrix of the distributions for each character probability
  %
  % OUTPUT:
  %   J   -- The cost for the current probability matrix.
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  J = -sum(log(sum(Y.*P, 1)));
endfunction
