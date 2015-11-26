function net = cnnff(net, x)
    n = numel(net.layers);
    %%x 28*28*50
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : n - 1  %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
				if strcmp(net.layers{l}.act, 'sigmoid')
                %  add bias, pass through nonlinearity
					net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
				elseif strcmp(net.layers{l}.act, 'tanh')
					net.layers{l}.a{j} = tanh(z + net.layers{l}.b{j});
				elseif strcmp(net.layers{l}.act, 'ReLU')
					net.layers{l}.a{j} = max(0, z + net.layers{l}.b{j});
				end;
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample 鍧囧?pooling,鍔犲叆dropout
			if strcmp(net.layers{l}.method, 'a')
                for j = 1 : inputmaps
					z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
					net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
					net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
					net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                end
			elseif strcmp(net.layers{l}.method, 'm')
                for j = 1 : inputmaps
					[net.layers{l}.a{j}, maxPosition] = MaxPooling(net.layers{l - 1}.a{j},[net.layers{l}.scale net.layers{l}.scale]);
					maxPosition = sparse(ones(length(maxPosition),1),maxPosition,ones(length(maxPosition),1),1,numel(net.layers{l - 1}.a{j}));
                    net.layers{l}.PosMatrix{j} = reshape(full(maxPosition),size(net.layers{l - 1}.a{j})); 
                    net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
                    net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                end
			elseif strcmp(net.layers{l}.method, 's')
                for j = 1 : inputmaps
					[net.layers{l}.a{j}, stoPosition] = StochasticPooling(net.layers{l - 1}.a{j},[net.layers{l}.scale net.layers{l}.scale]);
					stoPosition = sparse(ones(length(stoPosition),1),stoPosition,ones(length(stoPosition),1),1,numel(net.layers{l - 1}.a{j}));
                    net.layers{l}.PosMatrix{j} = reshape(full(stoPosition),size(net.layers{l - 1}.a{j})); 
                    net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
                    net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                end
			end
		end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n-1}.a)
        sa = size(net.layers{n-1}.a{j});
        net.fv = [net.fv; reshape(net.layers{n-1}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons,ffW鏉冮噸鐭╅樀,ffb鍋忕疆鍚戦噺
	if strcmp(net.layers{n}.form, 'log')
		net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
	elseif strcmp(net.layers{n}.form, 'softmax')
		M = net.ffW*net.fv;
        M = bsxfun(@plus, M, net.ffb);
        M = bsxfun(@minus, M, max(M, [], 1));
        M = bsxfun(@rdivide, exp(M), sum(exp(M)));
        net.o = M;
	end;
end
