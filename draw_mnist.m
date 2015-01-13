
function [] = draw_mnist(xs)
    l = size(xs)(1);

    for i = 0:(l-1)
        for j = 1:28
            for k = 1:28
                color = xs( (i+1), ((j-1) * 28 + k));
                rgbImage(i * 28 + j, k, :) = [color, color, color];
            end
        end
    end

    imshow(rgbImage);
    drawnow;
end
