
function [X, Y, y, N] = LoadAll(fileList)
  X = [];
  Y = [];
  y = [];
  for i=1:length(fileList)
  [Xi, Yi, yi, N] = LoadBatch(fileList{i});
  X = [X Xi];
  Y = [Y Yi];
  y = [y yi];
  endfor
  N = columns(X);
endfunction
