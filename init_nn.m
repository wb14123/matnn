
function nn = init_nn(layers)
	l = length(layers);

	for i = 1:(l-1)
		nn.activations{i} = zeros(1, layers(i));
		nn.weights{i} = rand(layers(i), layers(i+1));
		nn.bias{i} = rand(1, layers(i+1));
	end

	nn.activations{l} = zeros(1, layers(l));

	nn.sigmod = @sigmod;
	nn.sigmod_prime = @sigmod_prime;
end

function output = sigmod(input)
	output = 1 ./ (1 + exp(1) .^ (-input));
end

function output = sigmod_prime(input)
	output = sigmod(input) .* (1 - sigmod(input));
end
