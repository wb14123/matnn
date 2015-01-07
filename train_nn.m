
function [next_nn] = train_nn(nn, rate, xs, ys)

    figure(1);
    lHandle = line(nan, nan);
    
    for i = 1:100
        [nn, cost] = single_epoch(nn, 10, rate, xs, ys);
        
        X = get(lHandle, 'XData');
        Y = get(lHandle, 'YData');
        set(lHandle, 'XData', [X i], 'YData', [Y sum(cost)]);
        drawnow;
    end
   
    next_nn = nn;
end

function [next_nn, cost] = single_epoch(nn, batch_size, rate, xs, ys)
    l = length(xs);
    n = fix(l / batch_size);
    


    for i = 1:n
        weights = nn.weights;
        bias = nn.bias;
        for j = 0:(batch_size-1)
            [delta_weights, delta_bias, cost] = backprop(nn, rate, xs(i+j, :), ys(i+j, :));
            weights = sub_cell(weights, delta_weights, batch_size);
            bias = sub_cell(bias, delta_bias, batch_size);
        end
        
        nn.weights = weights;
        nn.bias = bias;
    end
    
    next_nn = nn;
end

function r = sub_cell(c1, c2, batch_size)
    r = cell(length(c1));
    for i = 1:length(c1)
        r{i} = c1{i} - c2{i} / batch_size;
    end
end
