function J = ComputeLoss(RNN, X, Y, h0, hp)
  % This function computes the loss based on the forward pass
  %
  % INPUT:
  %   RNN -- An octave structure containing all network parameters and index-maps
  %   X   -- The one-hot-representation of our input data
  %   Y   -- A one-hot-matrix on character level for the output data
  %   h0  -- an initial hidden state, h0
  %   hp  -- An octave structure containing all the hyper parameters
  %
  % OUTPUT:
  %   J   -- The loss for entire network
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  [P, H, J] = ForwardPass(RNN, X, Y, h0, hp);
endfunction
