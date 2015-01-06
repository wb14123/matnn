
function [next_nn] = train_nn(nn, rate, xs, ys)
	l = size(xs)(1);
	figure(1);
	lHandle = line(nan, nan);

	% for i = 1:10000
	% 	[nn, cost] = single_epoch(nn, 100, rate, xs, ys);

	% 	X = get(lHandle, 'XData');
	% 	Y = get(lHandle, 'YData');
	% 	set(lHandle, 'XData', [X i], 'YData', [Y sum(cost)]);
	% 	drawnow
	% end
	[nn, cost] = single_epoch(nn, 100, rate, xs, ys);

	next_nn = nn;

end

function [next_nn, cost] = single_epoch(nn, batch_size, rate, xs, ys)
	l = size(xs)(1);
	n = fix(l / batch_size);

	weights = nn.weights;
	bias = nn.bias;

	for i = 1:n
		for j = 0:(batch_size-1)
			[delta_weights, delta_bias, cost] = backprop(nn, rate, xs(i+j, :), ys(i+j, :));
			weights = sub_cell(weights, delta_weights);
			bias = sub_cell(bias, delta_bias);
		end
	end

	nn.weights = weights;
	nn.bias = bias;
	next_nn = nn;
end

function r = sub_cell(c1, c2)
	for i = 1:length(c1)
		r{i} = c1{i} - c2{i};
	end
end
