
function pr = validate_nn(nn, xs, ys)
    l = length(xs);

    right = 0;

    for i = 1:l
        next_nn = forward_nn(nn, xs(i, :));
        output = next_nn.activations{length(next_nn.activations)};
        cy = find(ys(i, :) == max(ys(i, :)));
        co = find(output == max(output));

        if (cy == co)
            right = right + 1;
        end
    end

    pr = right / l;
end
