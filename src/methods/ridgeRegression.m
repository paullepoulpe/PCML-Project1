function beta = ridgeRegression(y,tX, lambda)

% Define lambda matrix
lambdaMatrix = lambda*eye(size(tX,2));
lambdaMatrix(1,1) = 0;

% Compute beta
beta = inv(tX'*tX + lambdaMatrix)*tX'*y;

end