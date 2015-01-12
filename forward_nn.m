
function next_nn = forward_nn(nn, x)
    l = length(nn.activations);

    nn.zs = cell(1, l);
    nn.activations{1} = x;
    nn.zs{1} = x;
    for i = 2:l
        nn.zs{i} = nn.activations{i-1} * nn.weights{i-1} + nn.bias{i-1};
        nn.activations{i} = nn.sigmod(nn.zs{i});
    end
    next_nn = nn;
end
