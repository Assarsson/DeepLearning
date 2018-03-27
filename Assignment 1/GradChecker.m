function maxDiff = GradChecker(grad_analytic, grad_numeric)
  if (size(grad_analytic) != size(grad_numeric))
    maxDiff = 0;
    disp('Please input gradients with corresponding dimensions');
    return;
  endif
  difference = abs(grad_analytic .- grad_numeric)./max(1e-9, abs(grad_analytic).+abs(grad_numeric));
  maxDiff = max(max(difference));
  return;
endfunction
