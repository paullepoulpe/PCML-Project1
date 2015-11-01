function beta = penLogisticRegression(y,tX,alphaInitial,lambda)

% Maximum number of iterations
maxIters = 5000; % TO DEFINE

% Number of data
N = size(tX,1);

% alpha initialisation
alpha = alphaInitial;

% Beta initialisation
beta = zeros(size(tX,2),1);

% Iterate the gradient descent
for k = 1:maxIters
    
    % Compute the gradient
    g = computePenGradient(y, tX, beta, lambda);
    % Compute the Hessian
    H = computePenHessian(tX, beta, lambda);
    
    % Update beta using gradient descent
    % beta = beta - alpha.*g;
    % Update beta using Newton's method
    beta = beta - alpha.*(H^-1)*g;
    
    % Compute cost of logistic regression
    L = computeCostPenLogisticReg(y, tX, beta, lambda);
    alpha = abs(L*10^(-4));
    
    
    % Store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
    
    % Look at the convergence
    if k>1
%         fprintf('%d : %f  %f %f\n', k,L, alpha, max(abs(beta-beta_all(:,k-1))));
        if abs(beta-beta_all(:,k-1))<0.001
            % If the difference between last and present beta is small...
            % break
            break;
        end
    end
end

beta = beta_all(:,k);

end

function logLikelihood = computeCostPenLogisticReg(y, tX, beta, lambda)

logLikelihood = - sum( y.*(tX*beta) - log(ones(length(y),1) + exp(tX*beta))) ...
    +lambda*sum(beta.^2);

end
function gradient = computePenGradient(y, tX, beta, lambda)

gradient = tX'*(computeSigma(tX*beta)-y) + lambda*beta; % TO CHECK

end
function hessian = computePenHessian(tX, beta, lambda)

S = computeSigma(tX*beta).*(ones(size(tX,1),1)-computeSigma(tX*beta));
hessian = tX'*diag(S)*tX + lambda*ones(size(tX,2),size(tX,2)); % TO CHECK

end
function sigmaX = computeSigma(x)

sigmaX = exp(x)./(1+exp(x));

end