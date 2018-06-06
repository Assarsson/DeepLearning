function params = Generateparams(min_value, max_value, no_params)
  % Generateparams uniformly at random samples on the log scale given by the input.
  % it generates a variable amount of results and returns a vector containing these.
  %   min_value -- the lower log-bound of sampling
  %   max_value -- the upper log-bound of sampling
  %   no_params -- The number of parameters to sample
  %
  % OUTPUT:
  %   params -- A populated parameter vector of size (no_params, 1)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
params = 10.^(min_value+(max_value-min_value)*rand(no_params,1)');
endfunction
