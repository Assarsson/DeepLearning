function hp = GenerateHyperParameters(varargin)
  % This function initializes the network with hyper parameters that are not
  % included in the learning algorithm but the network. The size of the hidden state-layer,
  % the learning rate, the sequence length, the epsilon turn in the learning algo
  % and the number of epochs are such examples.
  %
  % INPUT:
  %   varagin -- A list of arguments that might be empty
  %
  % OUTPUT:
  %   hp  -- An octave structure containing the hyper parameters of the network
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  defaultParams = length(varargin) == 0;

  if (defaultParams)
    hp.m = 100;
    hp.eta = 0.1;
    hp.seqLength = 50;
    hp.epsilon = 1e-8;
    hp.epochs = 10;
  elseif (length(varargin) == 1)
    hp.m = varargin{1};
    hp.eta = 0.1;
    hp.seqLength = 25;
    hp.epsilon = 1e-8;
    hp.epochs = 10;
  elseif (length(varargin) == 2)
    hp.m = varargin{1};
    hp.eta = varargin{2};
    hp.seqLength = 25;
    hp.epsilon = 1e-8;
    hp.epochs = 10;
  else
    hp.m = varargin{1};
    hp.eta = varargin{2};
    hp.seqLength = varargin{3};
    hp.epsilon = 1e-8;
    hp.epochs = 10;
  end
endfunction
