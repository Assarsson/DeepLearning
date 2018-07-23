addpath Datasets/;
function bookData = LoadBatch(fileName)
  % This function acts as a trivial .txt-load interface
  % It takes in a filename, opens it as long as it's stored in /Datasets,
  % dumps the data to a variable
  % closes the file and returns the data.
  %
  % INPUT:
  %   fileName -- string variable containing the full filename
  %
  % OUTPUT:
  %   bookData -- A vector containing the entirety of the file
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  warning('off', 'all'); %Shuts down the filepath warnings
   %Adds the standard path for files
  fid = fopen(fileName, 'r'); %Opens the file into fid
  bookData = fscanf(fid, '%c'); %scans the entire file and dumps it to bookData
  fclose(fid); %Closes the file
endfunction %ends the function


function hp = GenerateHyperParameters(varargin)

  defaultParams = length(varargin) == 0;

  if (defaultParams)
    hp.m = 100;
    hp.eta = 0.1;
    hp.seqLength = 25;
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

function [RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx, ixToC, testing)
  sigma = 0.01;
  RNN.b = zeros(hp.m, 1);
  RNN.c = zeros(K, 1);
  RNN.U = randn(hp.m, K)*sigma;
  RNN.W = randn(hp.m, hp.m)*sigma;
  RNN.V = randn(K, hp.m)*sigma;
  RNN.K = K;
  RNN.m = hp.m;
  RNN.cToIx = cToIx;
  RNN.ixToC = ixToC;
  [~, RNN.N] = size(bookData);
  if (testing == true)
    RNN.N = 25;
  endif
  x0 = zeros(K, 1);
  h0 = zeros(hp.m, 1);
  X = zeros(RNN.K, RNN.N);
  Y = zeros(RNN.K, RNN.N);

  %convert to one-hot encoding
  Xchars = bookData(1:hp.seqLength);
  Ychars = bookData(2:hp.seqLength+1);
  for i = 1:RNN.N
    X(cToIx.(bookData(i)), i) = 1;
    Y(cToIx.(bookData(i)), i) = 1;
  endfor
endfunction


function [bookChars, cToIx, ixToC, K] = Preprocess(bookData)
  % This function acts as a preprocessing step for the data
  % It takes in the textdata and generates a vector of unique characters,
  % and two maps between the indices and the characters for later use.
  %
  % INPUT:
  %   bookData -- vector containing the full book
  %
  % OUTPUT:
  %   bookChars -- A vector containing the unique characters in the book
  %   xToIx -- a Map that maps from characters to their index
  %   ixToC -- a Map that maps from indices to their character.
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  bookChars = unique(bookData);
  cs = bookChars;
  K = length(bookChars);
  cToIx = struct();
  ixToC = struct();
  for c = cs
    ix = find(cs == c);
    cToIx.(c) = ix;
    ixToC.(num2str(ix)) = c;
  endfor
endfunction


function J = ComputeCost(Y, P)
  J = -sum(log(sum(Y.*P, 1)));
endfunction
function J = ComputeLoss(RNN, X, Y, h0, hp)
  [P, H, J] = ForwardPass(RNN, X, Y, h0, hp);
endfunction

function [P, H, J, hout] = ForwardPass(RNN, X, Y, h0, hp)
  P = zeros(hp.K, hp.seqLength);
  H = zeros(hp.m, hp.seqLength);
  h = h0;

  for t = 1:hp.seqLength
    x = X(:, t);
    at = RNN.W*h+ RNN.U*x + RNN.b;
    ht = tanh(at);
    ot = RNN.V*ht + RNN.c;
    pt = Softmax(ot);
    P(:, t) = pt;
    h = ht;
    H(:, t) = h;
  endfor
  hout = H(:,hp.seqLength);
  J = ComputeCost(Y, P);
endfunction


function gradients = BackwardPass(RNN, X, Y, P, H, hp, hprev)
  %initialize the matrices of intermediate vectorial gradients
  dO = zeros(hp.seqLength, hp.K); %for all ot's
  dH = zeros(hp.seqLength, hp.m); %for all ht's
  dA = zeros(hp.seqLength, hp.m); %for all at's

  %initialize a zeroed struct to hold the gradients as suggested in assignment
  fields = fieldnames(RNN);
  for field = 1:length(fields)
    gradients.(fields{field}) = zeros(size(RNN.(fields{field})));
  endfor

  % the V-gradient, c-gradient and intermediate o-gradients.
  for t = 1:hp.seqLength
    y = Y(:, t);
    p = P(:, t);
    h = H(:, t);
    g = -(y-p)';  %generate the g as per ushe.
    dO(t, :) = g; %save the ot's
    gradients.c += g';
    gradients.V += g'*h';
  endfor

  % The 'recursive' calculation of intermediate h and a-gradients.
  dH(hp.seqLength, :) = dO(hp.seqLength, :)*RNN.V;
  dA(hp.seqLength, :) = dH(hp.seqLength, :)*diag(1-(H(:, hp.seqLength)).^2);
  for t = hp.seqLength-1:-1:1
    dH(t, :) = dO(t,:)*RNN.V + dA(t + 1, :)*RNN.W;
    dA(t, :) = dH(t,:)*diag(1-H(:,t).^2);
  endfor

  % The W and U gradients that are left
  for t = 1:hp.seqLength
    gradients.U += dA(t, :)'*X(:, t)';
    gradients.b += dA(t, :)';
    if (t == 1)
      #FIXME: Understand why H(:,hp.seqLength) doesn't work. Might need new input/outputvariable
      gradients.W += dA(t, :)'*hprev';
      %gradients.W += dA(t, :)'*zeros(hp.m, 1)';
    else
      gradients.W += dA(t, :)'*H(:, t-1)';
    endif
  endfor
endfunction

function maxDiff = GradChecker(grad_analytic, grad_numeric)
  % Self implemented gradient checker. It follows the instruction from the lab
  % specification and computes the relative error between our gradient and the
  % numerically generated one.
  % INPUT:
  %   grad_analytic -- our calculated gradient of size (K, d)
  %   grad_numeric  -- Some numerically calculated gradient approximation (K, d)
  %
  % OUTPUT:
  %   maxDiff -- the largest recorded element-wise difference between our gradients.
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  difference = norm(grad_analytic - grad_numeric)./max(1e-6, norm(grad_analytic)+norm(grad_numeric));
  maxDiff = max(max(difference));
  return;
endfunction

function RNN = MiniBatchGD(X, RNN, hp)
  lossInit = true;
  smooth_loss = 0;
  count = 0;
  f = ['b', 'c', 'U', 'W', 'V'];
  f = cell(5,1);
  f(1,1) = 'b';
  f(2,1) = 'c';
  f(3,1) = 'U';
  f(4,1) = 'W';
  f(5,1) = 'V';
  f = f';
  for i = 1:length(f)
    m.(f{i}) = zeros(size(RNN.(f{i})));
  endfor

  for epoch = 1:hp.epochs
    e = 1;
    count = 1;
    hprev = zeros(RNN.m,1);
    loss_list = [];
    while (e+hp.seqLength < RNN.N)
      X_batch = X(:, e:e+hp.seqLength-1);
      Y_batch = X(:,e+1:e+hp.seqLength);
      [P, H, J, hout] = ForwardPass(RNN, X_batch, Y_batch, hprev, hp);
      gradients = BackwardPass(RNN, X_batch, Y_batch, P, H, hp, hprev);
      for i = 1:length(f)
        gradients.(f{i}) = max(min(gradients.(f{i}), 5), -5);
      end
      hprev = hout;
      for i = 1:length(f)
        m.(f{i}) += gradients.(f{i}).^2;
        RNN.(f{i}) = RNN.(f{i}) - hp.eta*gradients.(f{i})./sqrt(m.(f{i})+hp.epsilon);
      end
      if (count == 0)
        smooth_loss = J;
      else
        smooth_loss = .999*smooth_loss + 0.001*J;
      end
      e += hp.seqLength;
      if(mod(count, 500) == 0)
        msg = ['loss at epoch ' num2str(epoch) ' and iteration ' num2str(count)];
        disp(msg);
        smooth_loss
        X_batch = X(:,e:e+199);
        Y_batch = X(:,e+1:e+200);
        [P, H, Y] = Synthesize(RNN, hprev, X_batch, length(X_batch));
        for i = 1:200
          message(i) = RNN.ixToC.(num2str(find(Y(:,i) == 1)));
        end
        message
      endif
      loss_list = [loss_list smooth_loss];
      count += 1;
    end
    graphics_toolkit gnuplot;
    fig = figure();
    set(fig, 'visible', 'off');
    set(0,'defaultaxesfontname', 'Helvetica');
    hold on;
    titleText = ['Smooth loss function for epoch ' num2str(epoch)];
    disp(size(1:count));
    disp(size(loss_list));
    plot(1:count-1, loss_list, 'b');
    title(titleText);
    imageName = ['epoch ' num2str(epoch) '.jpg'];
    legend('smooth_loss');
    print(imageName);
  end
  [P, H, Y] = Synthesize(RNN, hprev, X_batch, 1000);
  for i = 1:1000
    message(i) = RNN.ixToC.(num2str(find(Y(:,i) == 1)));
  end
  message
endfunction

function p = Softmax(s)
  % This softmax implementation follow the standard scheme for such a function.
  % It's range is constrained to [0,1] and by summing over the exponents in the
  % denominator, we ensure a 'proper' distribution in a Kolmogorovian sense.
  % The commented line is to increase numerical stability once the gradients are
  % correct.
  % INPUT:
  %   s -- The result of an affine W*X+b-transformation of size (K, N)
  %
  % OUTPUT:
  %   p -- the probability matrix for the classes of X of size (K, N)
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  %s -= max(s);
  p = exp(s) ./ sum(exp(s));
endfunction

function [P, H, Y] = Synthesize(RNN, h0, x0, n)

  h = h0;
  x = x0(:,1);
  P = zeros(RNN.K, n); %Probability scores for the classes
  H = zeros(RNN.m, n); %intermittent hidden representations
  Y = zeros(RNN.K, n); %one-hot vector of output
  for t = 1:n
    at = RNN.W*h + RNN.U*x + RNN.b;
    ht = tanh(at);
    ot = RNN.V*ht + RNN.c;
    pt = Softmax(ot);
    %Implementation of suggested choice method
    cp = cumsum(pt);
    a = rand;
    ixs = find(cp-a > 0);
    ii = ixs(1);
    %add that choice to our output and use it for next timestep
    Y(ii, t) = 1;
    x = Y(:,t);
    %save our values and propagate hidden step.
    P(:, t) = pt;
    H(:, t) = ht;
    h = ht;
  endfor

endfunction
function num_grads = ComputeGradsNum(X, Y, RNN, h, hp)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h, hp);
end
endfunction

function grad = ComputeGradNum(X, Y, f, RNN, h, hp)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(RNN_try, X, Y, hprev, hp);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(RNN_try, X, Y, hprev, hp);
    grad(i) = (l2-l1)/(2*h);
end
endfunction

function allBookData = LoadAll(fileList)
  allBookData = [];
  for i=1:length(fileList)
    bookData = LoadBatch(fileList{i});
    allBookData = [allBookData bookData];
  endfor
endfunction







fileName = 'goblet_book.txt';
fileList = {'goblet_book.txt', 'azkaban_book.txt', 'phoenix_book.txt', 'halfblood_book.txt', 'hallows_book.txt'};
bookData = LoadAll(fileList);
disp('loaded book data');
msg = ['The number of characters in the dataset are: ' num2str(length(bookData))];
disp(msg);
hp = GenerateHyperParameters(400, 0.1, 25);
disp('generated hyper parameters');
[bookChars, cToIx, ixToC, K] = Preprocess(bookData);
disp('finished preprocessing data and creating index maps');
hp.K = K;
disp('Initializing parameters and one-hot-encoding dataset...');
[RNN, x0, h0, X, Y] = InitializeParameters(K, hp, bookData, cToIx, ixToC, testing = false);
disp('initialized network and parameters');
disp('running MiniBatch Gradient Descent');
RNN = MiniBatchGD(X, RNN, hp);
