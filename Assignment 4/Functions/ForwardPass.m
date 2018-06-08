function [P, H, J] = ForwardPass(RNN, X, Y, h0, hp)
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
    H(:, t) = h;
    h = ht;
  endfor

  J = ComputeCost(Y, P);
endfunction
