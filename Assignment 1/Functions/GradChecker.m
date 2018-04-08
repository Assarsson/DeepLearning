function maxDiff = GradChecker(grad_analytic, grad_numeric)
  % Self implemented gradient checker. It follows the instruction from the lab
  % specification and computes the relative error between our gradient and the
  % numerically generated one.
  % INPUT:
  %   grad_analytic -- our calculated gradient of size (K, d)
  %   grad_numeric  -- Some numerically calculated gradient approximation (K, d)
  %
  % OUTPUT:
  %   maxDiff -- the largest recorded element-wise difference between our gradients.
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

  if (size(grad_analytic) != size(grad_numeric))
    maxDiff = 0;
    disp('Please input gradients with corresponding dimensions');
    return;
  endif
  difference = norm(grad_analytic - grad_numeric)./max(1e-6, norm(grad_analytic)+norm(grad_numeric));
  maxDiff = max(max(difference));
  return;
endfunction
