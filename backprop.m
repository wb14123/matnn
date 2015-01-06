
function [next_nn, cost_v] = backprop(nn, rate, x, y)
	l = size(nn.activations)(2);

	% forward
	nn.activations{1} = x;
	for i = 2:l
		nn.activations{i} = nn.activations{i-1} * nn.weights{i-1} .+ nn.bias{i-1};
		nn.activations{i} = nn.sigmod(nn.activations{i});
	end

	% backprop
	cost{l} = cost_func(y, nn.activations{l});
	for i = 1:(l-1)
		j = l - i;

		cost{j} = cost{j+1} * transpose(nn.weights{j});
		d_bias = nn.sigmod_prime(cost{j + 1});
		d_weight = transpose(nn.activations{j}) * nn.sigmod_prime(cost{j + 1});

		nn.weights{j} = nn.weights{j} - (d_weight .* rate);
		nn.bias{j} = nn.bias{j} - (d_bias .* rate);
	end

	next_nn = nn;
	cost_v = cost{l};
end

function r = cost_func(y, a)
	r = (y - a) .^ 2;
end
