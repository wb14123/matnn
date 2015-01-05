
function [train_x, train_y, t10k_x, t10k_y] = read_mnist(mnist_dir)
	train_image_f = fopen(strcat(mnist_dir, '/train-images-idx3-ubyte'));
	train_label_f = fopen(strcat(mnist_dir, '/train-labels-idx1-ubyte'));
	t10k_image_f = fopen(strcat(mnist_dir, '/t10k-images-idx3-ubyte'));
	t10k_label_f = fopen(strcat(mnist_dir, '/t10k-labels-idx1-ubyte'));

	train_l = 60000;
	t10k_l = 10000;

	fread(train_image_f, 12);
	fread(train_label_f, 8);
	fread(t10k_image_f, 12);
	fread(t10k_label_f, 8);

	train_x = fread(train_image_f, [train_l 28*28]);
	train_y_inner = fread(train_label_f, [train_l 1]);
	train_y = zeros(train_l, 10);
	for n = 1:train_l
		train_y(n, cast(train_y_inner(n) + 1, 'int16')) = 1;
	end

	t10k_x = fread(t10k_image_f, [t10k_l 28*28]);
	t10k_y_inner = fread(t10k_label_f, [t10k_l 1]);
	t10k_y = zeros(t10k_l, 10);
	for n = 1:t10k_l
		train_y(n, cast(t10k_y_inner(n) + 1, 'int16')) = 1;
	end

	fclose(train_image_f);
	fclose(train_label_f);
	fclose(t10k_image_f);
	fclose(t10k_label_f);
end
