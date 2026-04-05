function run_training(cat_folder, non_cat_folder, output_csv, alpha, num_iters)
    % run_training: Antrenează o rețea neuronală pentru a recunoaște pisici
    % Argumente: cat_folder, non_cat_folder, output_csv, alpha, num_iters

    % 1. Încărcăm imaginile (folosind build_X cel nou și rapid)
    fprintf('Etapa 1: Încărcăm și procesăm imaginile...\n');
    
    X_cats = build_X(cat_folder);
    y_cats = ones(1, size(X_cats, 2)); % 1 pentru pisici
    
    X_non_cats = build_X(non_cat_folder);
    y_non_cats = zeros(1, size(X_non_cats, 2)); % 0 pentru restul
    
    % Combinăm datele într-o singură matrice X și un singur vector y
    X = [X_cats, X_non_cats];
    y = [y_cats, y_non_cats];
    
    % 2. Parametri pentru Rețeaua Neuronală (Neural Network)
    n_h = 64;     % numărul de neuroni în stratul ascuns
    lambda = 0.1; % parametrul de regularizare (L2)

    % 3. Antrenamentul propriu-zis (Apelăm noul train_model)
    fprintf('Etapa 2: Pornim motorul de antrenament (NN cu %d neuroni ascunsi)...\n', n_h);
    
    [W1, b1, W2, b2] = train_model(X, y, n_h, alpha, lambda, num_iters);

    % 4. Salvarea modelului
    % Salvăm greutățile învățate într-un fișier binar .mat (rapid și mic)
    save('model_final.mat', 'W1', 'b1', 'W2', 'b2', 'n_h');
    
    fprintf('Gata! Modelul a fost antrenat și salvat în model_final.mat\n');
    
    % 5. (Opțional) Afișăm acuratețea finală pe setul de antrenament
    % Aceasta este partea care îți va spune dacă ai depășit cei 58%
    Z1 = W1 * X + b1;
    A1 = max(0, Z1); % ReLU
    Z2 = W2 * A1 + b2;
    A2 = 1 ./ (1 + exp(-Z2)); % Sigmoid
    
    predictions = A2 > 0.5;
    accuracy = mean(double(predictions == y)) * 100;
    fprintf('Acuratețe pe setul de antrenament: %.2f%%\n', accuracy);
end