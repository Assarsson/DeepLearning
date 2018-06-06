function ng = BatchNormBackPass(g, s, mu, v, batchSize)
  ng = cell(batchSize, 1);
  epsilon = 1e-5;
  Vb = diag(v + epsilon);
  s = num2cell(s, 1);
  grad_v = cellfun(@(x, y) (y*Vb^(-3/2)*diag(x-mu))',s',g,'UniformOutput', false);
  grad_v = -1/2*sum([grad_v{:}], 2)';
  grad_mu = -sum([cellfun(@(x) (x*Vb^(-0.5))', g, 'UniformOutput', false){:}], 2);
  ng = cellfun(@(x, y) y*Vb^(-0.5)+2/batchSize*grad_v*diag(x-mu) + grad_mu'/batchSize, s',g, 'UniformOutput', false);
endfunction
