function net = cnnapplygrads(net, opts)
	mom = 0.5;
	net.iter = net.iter + 1;
	if net.iter == opts.momNew
		mom = opts.momNew;
	end
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
					net.layers{l}.vk{ii}{j} = mom * net.layers{l}.vk{ii}{j} + opts.alpha * net.layers{l}.dk{ii}{j}; 
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - net.layers{l}.vk{ii}{j};
                end
                net.layers{l}.vb{j} = mom * net.layers{l}.b{j} + opts.alpha * net.layers{l}.db{j};
				net.layers{l}.b{j} = net.layers{l}.b{j} - net.layers{l}.vb{j}
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW;
	net.vffW = net.ffW - net.vffW;
    net.ffb = net.ffb - opts.alpha * net.dffb;
	net.vffb = net.ffb - net.vffb;
end
