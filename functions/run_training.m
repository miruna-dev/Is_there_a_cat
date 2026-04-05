% --- Parametri noi ---
n_h = 64;     % numărul de neuroni în stratul ascuns
lambda = 0.1; % parametrul de regularizare

% --- Apelul noii funcții ---
[W1, b1, W2, b2] = train_model(X, y, n_h, alpha, lambda, num_iters);

% --- Salvare model ---
save('model_final.mat', 'W1', 'b1', 'W2', 'b2');