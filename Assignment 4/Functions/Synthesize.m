function [P, H, Y] = Synthesize(RNN, h0, x0, n)
  % This enveloping function performs the entire learning process end-to-end.
  % It trains over epochs, performs forward and backward passes,
  % utilizes the RMSProp optimization algorithm to descend over the gradients.
  % It then synthesizes, at every 500 iterations, a 200 character long text snippet.
  % Finally, it produces an error graph for each epoch for comparability.
  %
  % INPUT:
  %   RNN   -- An octave structure containing all network parameters and index-maps
  %   h0    -- An initial hidden state
  %   x0    -- An initial character
  %   n     -- The length of the synthesized text
  %
  % OUTPUT:
  %   P   -- Our final output probabilities
  %   H   -- Final hidden states
  %   Y   -- Our final one-hot encoded matrix of the generated text
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  h = h0;
  x = x0(:,1);
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
