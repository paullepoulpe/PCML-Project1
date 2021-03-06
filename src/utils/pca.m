function [ XReduced, V, VReduced ] = pca( X, minEigen)
% PCA Do PCA to remove the correlated dimensions of tX
% cov*V = V*lambda

covariance = cov(X'*X);
[V, lambda] = eig(covariance);

[~, lambdaCol] = find(lambda > minEigen);
VReduced = V(:,lambdaCol);

XProjected = (V*X')';
XReduced = (VReduced'*XProjected')';

end

