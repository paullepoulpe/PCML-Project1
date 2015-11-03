function [ trRMSE, teRMSE, lambdas ] = findBestRidgeLambda( X, y )
%FINDBESTRIDGELAMBDA

lambdas = logspace(-3, 7, 60);

predictParam = @(XTr, yTr, XTe, param) {
    predictRegression(XTr, yTr, XTe, @(XTr, yTr) ridgeRegression(XTr, yTr, param));
};


[trRMSE, teRMSE] = runRegressionParam(X, y, predictParam, lambdas);

end

