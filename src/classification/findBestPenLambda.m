function [ trRMSE, teRMSE, trLoss, teLoss, lambdas ] = findBestPenLambda( X, y )
%FINDBESTRIDGELAMBDA Summary of this function goes here
%   Detailed explanation goes here

lambdas = logspace(-3, 7, 60);

predictParam = @(XTr, yTr, XTe, param) {
    predictClassification(XTr, yTr, XTe, @(XTr, yTr) penLogisticRegression(XTr, yTr, 0.001, param));
};


[trRMSE, teRMSE, trLoss, teLoss] = runClassificationParam(X, y, predictParam, lambdas);

end

