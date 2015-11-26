function net = cnnbp(net, y)
    n = numel(net.layers);
	if strcmp(net.layers{n}.form, 'log')	
    %  error
		net.e = net.o - y;
    %  loss function
		net.L = gather(1/2* sum(net.e(:) .^ 2) / size(net.e, 2));
	%%  backprop delta	
		net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
	elseif strcmp(net.layers{n}.form, 'softmax')
	%  error
		net.e = (net.o - y) / size(net.layers{1}.a{1}, 3);
    %  loss function
		net.L = -1 * mean(sum(y.*log(net.o)));
    %%  backprop delta	
		net.od = net.e;
    end
	%% fvd:feature vector delta,特征向量误差
    net.fvd = (net.ffW' * net.od) * size(net.layers{1}.a{1}, 3);             
	%%如果输出层的前一层是Convnet,Convnet的输出需要经过activation处理
    if strcmp(net.layers{n}.type, 'c')         
        if strcmp(net.layers{n-1}.act, 'sigmoid')
			net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
		elseif strcmp(net.layers{n-1}.act, 'tanh')
			net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
		elseif strcmp(net.layers{n-1}.act, 'ReLU')
			net.fvd = net.fvd .* (net.fv > 0);
        end
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n-1}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n-1}.a)
        net.layers{n-1}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 2) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')	
            for j = 1 : numel(net.layers{l}.a)
				if strcmp(net.layers{l}.act ,'sigmoid')
					dc = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j});
				elseif strcmp(net.layers{l}.act, 'tanh')
					dc = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j});
				elseif strcmp(net.layers{l}.act, 'ReLU')
					dc = (net.layers{l}.a{j} > 0);
				end;
				%%average
                if strcmp(net.layers{l+1}.method, 'a')
					net.layers{l}.d{j} = dc .* (expand(net.layers{l + 1}.d{j} .* net.layers{l + 1}.dropoutMask{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
				%%maximum
				elseif strcmp(net.layers{l+1}.method, 'm')
					net.layers{l}.d{j} = dc .* (expand(net.layers{l + 1}.d{j} .* net.layers{l + 1}.dropoutMask{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) .* net.layers{l + 1}.PosMatrix{j});
				%%stochastic
				elseif strcmp(net.layers{l+1}.method, 's')
					net.layers{l}.d{j} = dc .* (expand(net.layers{l + 1}.d{j} .* net.layers{l + 1}.dropoutMask{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) .* net.layers{l + 1}.PosMatrix{j});
                end;
            end
        %%subsampling
            elseif strcmp(net.layers{l}.type, 's')
                for i = 1 : numel(net.layers{l}.a)
                    z = zeros(size(net.layers{l}.a{1}));
                     for j = 1 : numel(net.layers{l + 1}.a)
                        z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                     end
                        net.layers{l}.d{i} = z;
                end
            end
    end
      

    %%  calc gradients
    for l = 2 : n - 1 
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    %%尾部单层感知机的梯度
    if strcmp(net.layers{n}.form, 'log')
		net.dffW = net.od * (net.fv)' / size(net.od, 2);
		net.dffb = mean(net.od, 2);
	elseif strcmp(net.layers{n}.form, 'softmax')
		net.dffW = net.od * (net.fv)';
		net.dffb = sum(net.od, 2);
    end
	
 function x = rot180(x)
        x = flipdim(flipdim(x, 1), 2);
 end
end
