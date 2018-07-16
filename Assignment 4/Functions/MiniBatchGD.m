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
