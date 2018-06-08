function [P, H, Y] = Synthesize(RNN, h0, x0, n)

  h = h0;
  x = x0;
  P = zeros(RNN.K, n); %Probability scores for the classes
  H = zeros(RNN.m, n); %intermittent hidden representations
  Y = zeros(RNN.K, n); %one-hot vector of output
  for t = 1:n
    at = RNN.W*h + RNN.U*x + RNN.b;
    ht = tanh(at);
    ot = RNN.V*ht + RNN.c;
    pt = Softmax(ot);
    %Implementation of suggested choice method
    cp = cumsum(pt);
    a = rand;
    ixs = find(cp-a > 0);
    ii = ixs(1);
    %add that choice to our output and use it for next timestep
    Y(ii, t) = 1;
    x = Y(:,t);
    %save our values and propagate hidden step.
    P(:, t) = pt;
    H(:, t) = ht;
    h = ht;
  endfor

endfunction
