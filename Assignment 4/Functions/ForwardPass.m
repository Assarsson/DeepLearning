function [P, H, J, hout] = ForwardPass(RNN, X, Y, h0, hp)
  % This function computes the loss based on the forward pass
  %
  % INPUT:
  %   RNN   -- An octave structure containing all network parameters and index-maps
  %   X     -- The one-hot-representation of our input data
  %   Y     -- A one-hot-matrix on character level for the output data
  %   h0    -- an initial hidden state, h0
  %   hp    -- An octave structure containing all the hyper parameters
  %
  % OUTPUT:
  %   P     -- The probability matrix with the distribution for each character
  %   H     -- A matrix containing all the hidden states generated under the pass
  %   J     -- The scalar lass value for the network
  %   hout  -- the last generated hidden state to act as h0 in next iteration
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  P = zeros(hp.K, hp.seqLength);
  H = zeros(hp.m, hp.seqLength);
  h = h0;

  for t = 1:hp.seqLength
    x = X(:, t);
    at = RNN.W*h+ RNN.U*x + RNN.b;
    ht = tanh(at);
    ot = RNN.V*ht + RNN.c;
    pt = Softmax(ot);
    P(:, t) = pt;
    h = ht;
    H(:, t) = h;
  endfor
  hout = H(:,hp.seqLength);
  J = ComputeCost(Y, P);
endfunction
