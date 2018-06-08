function num_grads = ComputeGradsNum(X, Y, RNN, h, hp)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h, hp);
end

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
