function net = cnntrain(net, x, y, opts)
    %%x= 28*28*60000
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end

    net.rL = [];

    for i = 1 : opts.numepochs
        %disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        %tic;
      
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
           
            net = cnnff(net, batch_x);
          
            net = cnnbp(net, batch_y);
           %%梯度模型更新
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            if mod(l,10) == 0
				disp(['epoch' num2str(i) '/' num2str(opts.numepochs) 'batch' num2str(l) '/' num2str(numbatches)]);
            end
       % toc;
        end
	disp(['epoch' num2str(i) '/' num2str(opts.numepochs) 'error:' num2str(net.L)]);
    end
