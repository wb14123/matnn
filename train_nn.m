
function next_nn = train_nn(nn, rate, train_xs, train_ys, test_xs, test_ys)
    cost = [];
    pers = [];
    test_pers = [];

    train_xs = train_xs / 255;
    test_xs = test_xs / 255;

    figure;

    for i = 1:1000
        [next_nn, c] = single_epoch(nn, 1000, rate, train_xs, train_ys);
        nn = next_nn;
        cost = [cost ; c];

        % if (i > 1 && abs(cost(i-1) - cost(i)) < 0.01)
        %     break
        % end

        pers = [pers validate_nn(nn, train_xs, train_ys)];
        test_pers = [test_pers validate_nn(nn, test_xs, test_ys)];

        subplot(2, 3, 1);
        plot(cost);
        title('cost');

        subplot(2, 3, 2);
        plot(pers);
        title('pers');

        subplot(2, 3, 3)
        plot(test_pers);
        title('test pers');
        
        subplot(2, 3, 4);
        hist(cell2mat(cellfun(@(x)x(:),nn.weights(:),'un',0)));
        title('weights');
        
        subplot(2, 3, 5);
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

    c = zeros(1, length(ys(1, :)));

    for i = 1:n
        weights = nn.weights;
        bias = nn.bias;
        for j = 0:(batch_size-1)
            pos = (i-1) * batch_size + j + 1;
            [delta_weights, delta_bias, cost] = backprop(nn, rate, xs(pos, :), ys(pos, :));
            weights = sub_cell(weights, delta_weights, batch_size);
            bias = sub_cell(bias, delta_bias, batch_size);
            c = c + cost;
        end

        c = c / batch_size;
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
