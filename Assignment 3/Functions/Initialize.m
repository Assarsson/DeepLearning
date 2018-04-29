function [W, b] = Initialize(d, layerData, initType)
  % Initialize creates our W and b cells and populates them with random values.
  % It either utilizes a random gaussian prior on the parameters or a Xavier prior.
  % The role of the Xavier prior is to keep X and and W*X equivariant, to increase
  % stability of the model and increase speed in our iterations.
  % INPUT:
  %   d -- The dimensionality of our input X.
  %   layerData -- A vector containing the number of nodes in each layer, including last.
  %   initType -- Optional variable that can invoke xavier-initialization.
  %
  % OUTPUT:
  %   W -- A populated weight cell array of size (layers, 1)
  %   b -- A populated bias cell array of size (layers, 1)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  mu = 0;
  variance = 0.000001;
  layers = length(layerData);
  W = cell(layers,1);
  b = cell(layers,1);

  W1 = double(randn(layerData(1), d));
  b1 = double(zeros(layerData(1), 1));
  W(1, 1) = W1;
  b(1, 1) = b1;
  for layer = 2:layers
    Weight = double(randn(layerData(layer), layerData(layer-1)));
    bias = double(zeros(layerData(layer), 1));
    W(layer, 1) = Weight;
    b(layer, 1) = bias;
  endfor
  if (initType == 'xavier')
    W = cellfun(@(x) x*sqrt(1/length(x)) + mu, W, 'UniformOutput', false);
    return;
  end
  W = cellfun(@(x) x*sqrt(variance) + mu, W, 'UniformOutput', false);
endfunction
