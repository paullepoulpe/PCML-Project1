function [ XReduced, V, VReduced ] = pca( X, minEigen)
% PCA Do PCA to remove the correlated dimensions of tX

covariance = cov(X'*X);
[V, lambda] = eig(covariance);
[lambdaRow, lambdaCol] = find(lambda > minEigen);

% cov*V = V*lambda

VReduced = V(:,lambdaCol(1):end);
tXProjected = (V*X')';
XReduced = (VReduced'*tXProjected')';

end

