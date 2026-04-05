function [W1, b1, W2, b2] = gradient_descent(X, y, W1, b1, W2, b2, alpha, lambda, num_iters)
    m = size(X, 2); % numărul de exemple (coloane)

    for i = 1:num_iters
        % --- FORWARD PROPAGATION ---
        Z1 = W1 * X + b1;
        A1 = max(0, Z1); % ReLU Activation
        
        Z2 = W2 * A1 + b2;
        A2 = 1 ./ (1 + exp(-Z2)); % Sigmoid Activation (Predicția finală)

        % --- BACKPROPAGATION ---
        % Eroarea la nivelul stratului de ieșire
        dZ2 = A2 - y;
        dW2 = (1/m) * (dZ2 * A1') + (lambda/m) * W2;
        db2 = (1/m) * sum(dZ2, 2);

        % Eroarea la nivelul stratului ascuns
        dA1 = W2' * dZ2;
        dZ1 = dA1 .* (Z1 > 0); % Derivata ReLU
        dW1 = (1/m) * (dZ1 * X') + (lambda/m) * W1;
        db1 = (1/m) * sum(dZ1, 2);

        % --- UPDATE PARAMETERS ---
        W1 = W1 - alpha * dW1;
        b1 = b1 - alpha * db1;
        W2 = W2 - alpha * dW2;
        b2 = b2 - alpha * db2;

        % Afișăm costul la fiecare 100 de iterații
        if mod(i, 100) == 0
            cost = (-1/m) * sum(y .* log(A2 + 1e-15) + (1-y) .* log(1-A2 + 1e-15));
            fprintf('Iterația %d, Cost: %.4f\n', i, cost);
        end
    end
end