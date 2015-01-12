
function [delta_weights, delta_bias, cost_v] = backprop(nn, rate, x, y)
    l = length(nn.activations);

    % forward
    nn = forward_nn(nn, x);

    % backprop
    cost{l} = (nn.activations{l} - y);

    delta_bias = cell(l-1);
    delta_weights = cell(l-1);

    for i = 1:(l-1)
        j = l - i;

        cost{j} = cost{j+1} * transpose(nn.weights{j}) .* nn.sigmod_prime(nn.zs{j});
        delta_bias{j} = cost{j+1} * rate;
        delta_weights{j} = transpose(nn.activations{j}) * cost{j+1} * rate;
    end

    cost_v = -y .* log(nn.activations{l}) - (1-y) .* log(1 - nn.activations{l});
end
