function beta = leastSquares(y,tX)

% Compute beta
beta = (tX'*tX)^(-1)*(tX'*y);

end