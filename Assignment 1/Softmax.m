function p = Softmax(s)
  s -= max(s);
  p = exp(s) ./ sum(exp(s));
endfunction
