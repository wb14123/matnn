
function nn = init_nn(layers)
    l = length(layers);

    for i = 1:(l-1)
        nn.activations{i} = zeros(1, layers(i));
        nn.weights{i} = randn(layers(i), layers(i+1));
        nn.bias{i} = randn(1, layers(i+1));
    end

    nn.activations{l} = zeros(1, layers(l));

    nn.sigmod = @sigmod;
    nn.sigmod_prime = @sigmod_prime;
end

function output = sigmod(input)
    output = 1 ./ (1 + exp(-input));
end

function output = sigmod_prime(input)
    output = sigmod(input) .* (1 - sigmod(input));
end
