function gradients = BackwardPass(RNN, X, Y, P, H, hp, hprev)
    % This function computes the gradients of the network in our
    % backward update-pass. It takes in a RNN-structure, the input-data
    % the shifted Y-data, a matrix of forwardpass probabilities, a matrix of
    % the hidden states in the network, a parameter object and the last hidden state
    % It initializes gradient parameters, computes the derivative of cross-entropy-loss
    % and updates the final parameters, V and c from oht = Vht + c
    % It then propagates through the non-linearity ht = tanh(at),
    % and finally calculates the gradients for the at = Wht-1 + Uxt + b layer.
    %
    % INPUT:
    %   RNN   -- An octace structure containing parameters and index-conversion-maps.
    %   X     -- A one-hot-matrix on character level for the input data
    %   Y     -- A one-hot-matrix on character level for the output data
    %   P     -- A probability matrix of the distributions for each character probability
    %   H     -- a matrix containing the m hidden states of the network passed forward
    %   hp    -- An octave structure containing hyper parameters external to the learning algorithm
    %   hprev -- The initialization hidden state, h0
    %
    % OUTPUT:
    %   gradients -- An octave structure containing the gradients for V, c, W, U, b
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

  %initialize the matrices of intermediate vectorial gradients
  dO = zeros(hp.seqLength, hp.K); %for all ot's
  dH = zeros(hp.seqLength, hp.m); %for all ht's
  dA = zeros(hp.seqLength, hp.m); %for all at's

  %initialize a zeroed struct to hold the gradients as suggested in assignment
  fields = fieldnames(RNN);
  for field = 1:length(fields)
    gradients.(fields{field}) = zeros(size(RNN.(fields{field})));
  endfor

  % the V-gradient, c-gradient and intermediate o-gradients.
  for t = 1:hp.seqLength
    y = Y(:, t);
    p = P(:, t);
    h = H(:, t);
    g = -(y-p)';  %generate the g as per ushe.
    dO(t, :) = g; %save the ot's
    gradients.c += g';
    gradients.V += g'*h';
  endfor

  % The 'recursive' calculation of intermediate h and a-gradients.
  dH(hp.seqLength, :) = dO(hp.seqLength, :)*RNN.V;
  dA(hp.seqLength, :) = dH(hp.seqLength, :)*diag(1-(H(:, hp.seqLength)).^2);
  for t = hp.seqLength-1:-1:1
    dH(t, :) = dO(t,:)*RNN.V + dA(t + 1, :)*RNN.W;
    dA(t, :) = dH(t,:)*diag(1-H(:,t).^2);
  endfor

  % The W and U gradients that are left
  for t = 1:hp.seqLength
    gradients.U += dA(t, :)'*X(:, t)';
    gradients.b += dA(t, :)';
    if (t == 1)
      gradients.W += dA(t, :)'*hprev';
    else
      gradients.W += dA(t, :)'*H(:, t-1)';
    endif
  endfor
endfunction
