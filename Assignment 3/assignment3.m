addpath Functions/

layerData = [50,10];
[W, b] = Initialize(12, layerData, 'gaussi');
disp(var(W{1,1}));
