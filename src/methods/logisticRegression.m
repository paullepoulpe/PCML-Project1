function beta = logisticRegression( y, tX, alphaInitial )

% Maximum number of iterations
maxIters = 15000;

% Number of data
N = size(tX,1);

% alpha initialisation
alpha = alphaInitial;

% Beta initialisation
beta = zeros(size(tX,2),1);

% Iterate the gradient descent
for k = 1:maxIters
    % Compute the gradient
    g = computeGradient(y, tX, beta);
    % Compute the Hessian
    H = computeHessian(tX, beta);
    
    % Update beta using Newton's method
    beta = beta - alpha.*(H^-1)*g;
    
    % Compute cost of logistic regression
    L = computeCostLogisticReg(y, tX, beta);
    
    % Break if the algorithm has diverged during the current iteration
    if L == Inf
        fprintf('Cost explosion, last iteration taken\n');
        beta_all(:,k) = beta_all(:,k-1);
        break;
    end
    
    % Compute step size
    alpha = abs(L*10^(-4));
    
    % Store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
    
    % Look at the convergence
    if k>1
        if abs(beta-beta_all(:,k-1))<0.001
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