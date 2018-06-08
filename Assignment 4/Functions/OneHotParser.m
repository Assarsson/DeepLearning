function text = OneHotParser(Y, IxToC)
  text = [];
  [ixs,_] = find(Y == 1);
  for ix = ixs'
    text = [text IxToC.(num2str(ix))];
  endfor
endfunction
