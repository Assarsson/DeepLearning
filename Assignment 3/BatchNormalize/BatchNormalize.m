function s = BatchNormalize(s, mu, v)
%
%
% INPUT:
%   s  -- the output from our linear transformation at each layer
%   mu -- estimated mean for the unnormalized s-value
%   v  -- vector of estimated variance for each dimension of s
%
% OUTPUT:
%   s  -- the batch normalized version of input-s.
%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
epsilon = 1e-5;
s = 1./sqrt(diag(v + epsilon))*(s-mu);
endfunction
