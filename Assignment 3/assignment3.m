addpath Functions/

layerData = [5,10];
[W, b] = Initialize(5, layerData, 'gaussi');
[Wm, bm] = InitializeMomentum(W, b);
