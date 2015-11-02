function [ trRMSE, teRMSE, lambdas ] = findBestGDAlpha( X, y )
%FINDBESTRIDGELAMBDA Summary of this function goes here
%   Detailed explanation goes here

lambdas = logspace(-3, -0.5, 20);

predictParam = @(XTr, yTr, XTe, param) {
    predictRegression(XTr, yTr, XTe, @(XTr, yTr) leastSquaresGD(XTr, yTr, param));
};


[trRMSE, teRMSE] = runRegressionParam(X, y, predictParam, lambdas);

end

