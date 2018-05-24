function g = BatchNormBackPass(g, s,si, mu, v, batchSize)

  epsilon = 1e-3;
  Vb = diag(v + epsilon);
  s = num2cell(s, 1);
  grad_v = cellfun(@(x) (g*Vb^(-3/2)*diag(x-mu))',s,'UniformOutput', false);
  grad_v = -1/2*sum([grad_v{:}], 2)';
  grad_mu = -sum(g*Vb^(-0.5));
  g = g*Vb^(-0.5)+2/batchSize*grad_v*diag(si-mu) + 1/batchSize*grad_mu;
endfunction
