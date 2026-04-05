function run_training(cat_folder, non_cat_folder)
    % --- PARAMETRI ---
    alpha = 0.005;     % Learning rate (mai mic e mai sigur)
    lambda = 0.1;      % Regularizare
    num_iters = 1000;
    n_h = 64;          % Număr de neuroni în stratul ascuns
    
    fprintf('Încarc și procesez imaginile (fără CSV!)...\n');
    % Folosim build_X-ul cel nou care face și Data Augmentation
    X_cats = build_X(cat_folder);
    y_cats = ones(1, size(X_cats, 2));
    
    X_non_cats = build_X(non_cat_folder);
    y_non_cats = zeros(1, size(X_non_cats, 2));
    
    % Combinăm datele
    X = [X_cats, X_non_cats];
    y = [y_cats, y_non_cats];
    
    % Amestecăm datele (Shuffle) - Vital pentru antrenament!
    p = randperm(size(X, 2));
    X = X(:, p);
    y = y(:, p);

    fprintf('Încep antrenamentul pe un strat ascuns de %d neuroni...\n', n_h);
    [W1, b1, W2, b2] = train_model(X, y, n_h, alpha, lambda, num_iters);

    % Calculăm acuratețea pe setul de antrenament
    Z1 = W1 * X + b1;
    A1 = max(0, Z1);
    Z2 = W2 * A1 + b2;
    A2 = 1 ./ (1 + exp(-Z2));
    predictions = A2 > 0.5;
    
    accuracy = mean(double(predictions == y)) * 100;
    fprintf('Gata! Acuratețe antrenament: %.2f%%\n', accuracy);
    
    % Salvăm modelul eficient în format MATLAB binar
    save('model_pisici.mat', 'W1', 'b1', 'W2', 'b2', 'n_h');
    fprintf('Modelul a fost salvat în model_pisici.mat\n');
end