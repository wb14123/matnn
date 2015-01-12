
function [next_nn] = train_nn(nn, rate, train_xs, train_ys, test_xs, test_ys)
    cost = [];
    pers = [];

    train_xs = train_xs / 255 - 0.5;
    test_xs = test_xs / 255 - 0.5;
    
    figure;

    for i = 1:100
        [next_nn, c] = single_epoch(nn, 1000, rate, train_xs, train_ys);
        nn = next_nn;
        cost = [cost mean(c)];
        display(cost);

        pers = [pers validate_nn(nn, test_xs, test_ys)];
        display(pers);
        
        subplot(2, 2, 1);
        plot(cost);
        title('cost');
        
        subplot(2, 2, 2);
        plot(pers);
        title('pers');
        
        subplot(2, 2, 3);
        hist(cell2mat(cellfun(@(x)x(:),nn.weights(:),'un',0)));
        title('weights');
        
        subplot(2, 2, 4);
        hist(cell2mat(cellfun(@(x)x(:),nn.bias(:),'un',0)))
        title('bias');
       
        
        drawnow;
    end

    next_nn = nn;
end

function [next_nn, c] = single_epoch(nn, batch_size, rate, xs, ys)
    l = length(xs);
    n = fix(l / batch_size);

    index = randperm(l);
    xs = xs(index, :);
    ys = ys(index, :);

    c = zeros(1, n);

    for i = 1:n
        weights = nn.weights;
        bias = nn.bias;
        for j = 0:(batch_size-1)
            pos = (i-1) * batch_size + j + 1;
            [delta_weights, delta_bias, cost] = backprop(nn, rate, xs(pos, :), ys(pos, :));
            weights = sub_cell(weights, delta_weights, batch_size);
            bias = sub_cell(bias, delta_bias, batch_size);
            c(i) = c(i) + sum(cost);
        end

        c(i) = c(i) / batch_size;
        nn.weights = weights;
        nn.bias = bias;
    end
    next_nn = nn;
end

function r = sub_cell(c1, c2, batch_size)
    r = cell(1, length(c1));
    for i = 1:length(c1)
        r{i} = c1{i} - c2{i} / batch_size;
    end
end
