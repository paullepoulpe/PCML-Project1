function beta = logisticRegression(y,tX,alpha)

% Maximum number of iterations
maxIters = 10000; % TO DEFINE

% Number of data
N = size(tX,1);

% Beta initialisation
beta = zeros(size(tX,2),1);

% Iterate the gradient descent
for k = 1:maxIters
    % Compute cost of logistic regression
    L = computeCostLogisticReg(y, tX, beta);
    % Compute the gradient
    g = computeGradient(y, tX, beta);
    % Compute the Hessian
    H = computeHessian(tX, beta);
    
    % Update beta using gradient descent
    beta = beta - alpha.*g;
    % Update beta using Newton's method
    % beta = beta - alpha.*(H^-1)*g;
    
    % Store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
    
    
    
    % Look at the convergence
    if k>1
        fprintf('%d : %f\n', k, max(abs(beta-beta_all(k-1))));
        if abs(beta-beta_all(k-1))<0.01
            % If the difference between last and present beta is small...
            % break
            break;
        end
    end
end

beta = beta_all(:,k);

end

function logLikelihood = computeCostLogisticReg(y, tX, beta)

logLikelihood = - sum( y.*(tX*beta) - log(ones(length(y),1) + exp(tX*beta)));

end
function gradient = computeGradient(y, tX, beta)

gradient = tX'*(computeSigma(tX*beta)-y);

end
function hessian = computeHessian(tX, beta)

S = computeSigma(tX*beta).*(ones(size(tX,1),1)-computeSigma(tX*beta));
hessian = tX'*diag(S)*tX;

end
function sigmaX = computeSigma(x)

sigmaX = exp(x)./(1+exp(x));

end