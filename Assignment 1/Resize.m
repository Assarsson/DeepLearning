function [X,Y,y,W,N] = Resize(X,Y,y,W, size = 100, dimension = 3072)
  X = X(1:dimension,1:size);
  Y = Y(:,1:size);
  y = y(:,1:size);
  W = W(:,1:dimension);
  N = size;
  return
endfunction
