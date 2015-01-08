
function [delta_weights, delta_bias, cost_v] = backprop(nn, rate, x, y)
	l = length(nn.activations);
    
    x = x / norm(x);
   

	% forward
    nn.zs = cell(l);
	nn.activations{1} = x;
    nn.zs{1} = x;
	for i = 2:l
		nn.zs{i} = nn.activations{i-1} * nn.weights{i-1} + nn.bias{i-1};
		nn.activations{i} = nn.sigmod(nn.zs{i});
	end

	% backprop
	cost{l} = (nn.activations{l} - y) .* nn.sigmod_prime(nn.zs{l});
    
    delta_bias = cell(l-1);
    delta_weights = cell(l-1);
    
	for i = 1:(l-1)
		j = l - i; 

		cost{j} = cost{j+1} * transpose(nn.weights{j}) .* nn.sigmod_prime(nn.zs{j});
		delta_bias{j} = cost{j+1} * rate;
		delta_weights{j} = transpose(nn.activations{j}) * cost{j+1} * rate;
	end

	cost_v = (nn.activations{l} - y) .^ 2;
end


