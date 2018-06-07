function hp = GenerateHyperParameters(varargin)

  defaultParams = length(varargin) == 0;

  if (defaultParams)
    hp.m = 100;
    hp.eta = 0.1;
    hp.seqLength = 25;
  elseif (length(varargin) == 1)
    hp.m = varargin{1};
    hp.eta = 0.1;
    hp.seqLength = 25;
  elseif (length(varargin) == 2)
    hp.m = varargin{1};
    hp.eta = varargin{2};
    hp.seqLength = 25;
  else
    hp.m = varargin{1};
    hp.eta = varargin{2};
    hp.seqLength = varargin{3};
  end
endfunction
