function beta = leastSquaresGD(y,tX,alpha)

% Maximum number of iterations
maxIters = 1000; % TO DEFINE
epsilon = 0.01;

% Number of data
N = size(tX,1);

% Beta initialisation
beta = zeros(size(tX,2),1);

% Iterate the gradient descent
for k = 1:maxIters
    % Compute the error
    e = y - tX*beta;
    % Compute the gradient for least squares
    g = -1/N * tX'*e;
    % Compute MSE
    L = e'*e/(2*N);
    
    % Update beta using gradient descent
    beta = beta - alpha.*g;
    
    % Store beta and L 
    beta_all(:,k) = beta; % FOR DEBUG
    L_all(k) = L; % FOR DEBUG
    
    % Look at the convergence
    if k>1
        if abs(beta-beta_all(k-1)) < epsilon
            % If the difference between last and present beta is small...
            % break
            fprintf('Converged !\n'); 
            break;
        end
    end
end

if abs(beta-beta_all(k-1)) >= epsilon
    fprintf('Reached end of iterations\n');
end

% Final value of beta
beta = beta_all(:,k);

end