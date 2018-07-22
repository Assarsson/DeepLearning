%Set up and initialization
addpath Functions/
addpath Helpfunctions/
fileName = 'goblet_book.txt';
bookData = LoadBatch(fileName);
hp = GenerateHyperParameters(5);
[bookChars, cToIx, ixToC, K] = Preprocess(bookData);
hp.K = K;
[RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx, ixToC, testing = true);
hprev = h0;
[P, H, J, hout] = ForwardPass(RNN, X, Y, hprev, hp);
gradients = BackwardPass(RNN, X, Y, P, H, hp, hprev);
hprev = hout;
num_gradients = ComputeGradsNum(X, Y, RNN, h=1e-4, hp);
fields = fieldnames(gradients);
for i = 1:length(fields)
  disp(GradChecker(gradients.(fields{i}), num_gradients.(fields{i})));
endfor
