function [W1, b1, W2, b2] = train_model(X, y, n_h, alpha, lambda, num_iters)
    % n_h = numărul de neuroni în stratul ascuns (încearcă 64 sau 128)
    [n_x, m] = size(X);
    n_y = 1; % Un singur output (Pisică/Nu)

    % He Initialization pentru W1 (bună pentru ReLU)
    W1 = randn(n_h, n_x) * sqrt(2/n_x);
    b1 = zeros(n_h, 1);
    
    % Xavier Initialization pentru W2 (bună pentru Sigmoid)
    W2 = randn(n_y, n_h) * sqrt(1/n_h);
    b2 = zeros(n_y, 1);

    % Pornim antrenamentul
    [W1, b1, W2, b2] = gradient_descent(X, y, W1, b1, W2, b2, alpha, lambda, num_iters);
end