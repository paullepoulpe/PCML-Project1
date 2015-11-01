function [ XReduced, V, VReduced ] = pca( X, minEigen)
% PCA Do PCA to remove the correlated dimensions of tX
% cov*V = V*lambda
normX = normalise(X'*X);
covariance = cov(normX);
[V, lambda] = eig(covariance);

[~, lambdaCol] = find(lambda > minEigen);
VReduced = V(:,lambdaCol);

XProjected = (V*X')';
XReduced = (VReduced'*XProjected')';

end

