function J = ComputeLoss(RNN, X, Y, h0, hp)
  [P, H, J] = ForwardPass(RNN, X, Y, h0, hp);
endfunction
