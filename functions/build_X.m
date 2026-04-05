function X = build_X(folder_path)
    % Citim toate imaginile .jpg din folder
    extensii = {'*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png'};
    files = [];
    for i = 1:length(extensii)
        files = [files; dir(fullfile(folder_path, extensii{i}))];
    end
m = length(files);
    m = length(files); 
    
    % Setăm rezoluția dorită: 224x224
    target_size = [224, 224];
    
    % Numărul de pixeli dintr-o imagine alb-negru (224 * 224 = 50176)
    n = target_size(1) * target_size(2);
    
    % --- DATA AUGMENTATION ---
    % Deoarece pentru fiecare imagine adăugăm și varianta ei în oglindă,
    % numărul total de coloane din X va fi dublu față de câte poze ai în folder.
    total_images = m * 2;
    
    % Inițializăm X pentru alocare de memorie (extrem de important pentru viteză)
    X = zeros(n, total_images);
    
    idx = 1; % Un index ca să știm pe ce coloană din X scriem
    
    for i = 1:m
        % Generăm calea către imagine
        img_path = fullfile(folder_path, files(i).name);
        
        % 1. Citim imaginea nativ cu MATLAB
        img = imread(img_path);
        
        % 2. Transformăm în Alb-Negru (dacă imaginea e color are 3 canale)
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % 3. Redimensionăm imaginea la 224x224
        img_resized = imresize(img, target_size);
        
        % 4. Normalizăm valorile pixelilor (de la 0-255 la 0.0-1.0)
        % Folosim double() pentru că rețelele neuronale au nevoie de zecimale
        img_normalized = double(img_resized) / 255.0;
        
        % 5. Vectorizăm matricea imaginii. 
        % Sintaxa (:) ia matricea 2D și o transformă automat într-un vector coloană lung.
        x_original = img_normalized(:);
        
        % Salvăm imaginea originală pe coloana curentă și creștem indexul
        X(:, idx) = x_original;
        idx = idx + 1;
        
        % 6. Creăm imaginea în oglindă (Flip Left-Right)
        img_flipped = fliplr(img_normalized);
        x_flipped = img_flipped(:);
        
        % Salvăm și clona în oglindă pe următoarea coloană
        X(:, idx) = x_flipped;
        idx = idx + 1;
        
        % (Opțional) Un mic print ca să vezi că nu s-a blocat codul
        if mod(i, 50) == 0
            disp(['Am procesat ', num2str(i), ' din ', num2str(m), ' imagini originale...']);
        end
    end
    
    disp('Gata! Matricea X a fost construită.');
    disp(['Dimensiunea finală a lui X: ', num2str(n), ' pixeli pe linie x ', num2str(total_images), ' imagini pe coloană.']);
end