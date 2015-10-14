function beta  =  ridgeRegression(y,tX,lambda)

lambdaMatrix = lambda*eye(size(tX,2));
lambdaMatrix(1,1) = 0;

beta = inv(tX'*tX + lambdaMatrix)*tX'*y;
%beta = (tX'*tX)^(-1)*(tX'*y);

end