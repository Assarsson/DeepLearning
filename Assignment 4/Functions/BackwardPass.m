function gradients = BackwardPass(RNN, X, Y, P, H, hp)
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
      #FIXME: Understand why H(:,hp.seqLength) doesn't work. Might need new input/outputvariable
      %gradients.W += dA(t, :)'*H(:,hp.seqLength)';
      gradients.W += dA(t, :)'*zeros(hp.m, 1)';
    else
      gradients.W += dA(t, :)'*H(:, t-1)';
    endif
  endfor
endfunction
