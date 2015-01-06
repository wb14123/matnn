
function [next_nn] = train_nn(nn, rate, xs, ys)
	l = size(xs)(1);
	figure(1);
	lHandle = line(nan, nan);

	for i = 1:l
		[nn, cost] = backprop(nn, rate, xs(i, :), ys(i, :));
		X = get(lHandle, 'XData');
		Y = get(lHandle, 'YData');
		set(lHandle, 'XData', [X i], 'YData', [Y sum(cost)]);
		drawnow
	end

	next_nn = nn;

end
