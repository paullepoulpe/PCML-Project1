function beta = leastSquares(y,tX)

beta = (tX'*tX)^(-1)*(tX'*y);

end