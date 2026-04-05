function J = compute_cost(a, y, theta, lambda)
  % a = probabilitățile prezise (vector 1 x m)
  % y = etichetele reale (vector 1 x m)
  % theta = greutățile modelului (necesare pentru regularizare)
  % lambda = parametrul de regularizare (ex: 0.1 sau 1)

  m = length(y);
  
  % --- 1. Fixăm problema log(0) ---
  % Dacă a este exact 0 sau 1, log(a) dă minus infinit și strică tot.
  % "Clipuim" valorile să fie foarte aproape de 0 sau 1, dar nu fix acolo.
  epsilon = 1e-15;
  a = max(epsilon, min(1 - epsilon, a));

  % --- 2. Costul de bază (Cross-Entropy) ---
  core_cost = (-1/m) * sum(y .* log(a) + (1 - y) .* log(1 - a));

  % --- 3. Termenul de Regularizare L2 ---
  % Acesta penalizează valorile theta prea mari. 
  % "Îmblânzește" modelul ca să nu devină prea specific pentru pozele de antrenament.
  % Excludem theta(1) dacă îl folosești ca Bias (opțional, dar recomandat)
  theta_rest = theta(2:end); 
  regularization = (lambda / (2 * m)) * sum(theta_rest .^ 2);

  % Costul final
  J = core_cost + regularization;
end