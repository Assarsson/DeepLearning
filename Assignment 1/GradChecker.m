function maxDiff = GradChecker(grad_analytic, grad_numeric)
  if (size(grad_analytic) != size(grad_numeric))
    maxDiff = 0;
    disp('Please input gradients with corresponding dimensions');
    return;
  endif
  disp('analytic gradient: '),disp(grad_analytic(:,1));
  disp('numeric gradient: '),disp(grad_numeric(:,1));
  disp(mean(mean(abs(grad_analytic .- grad_numeric))));
  difference = abs(grad_analytic .- grad_numeric)./max(1e-6, abs(grad_analytic).+abs(grad_numeric));
  maxDiff = max(max(difference));
  return;
endfunction
