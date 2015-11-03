function [ trRMSE, teRMSE, trLoss, teLoss, lambdas ] = findBestPenLambda( X, y )
%FINDBESTRIDGELAMBDA Find the optimal lambda to minimise the error of the
%penalised logistic regression

lambdas = logspace(-3, 0, 10);

predictParam = @(XTr, yTr, XTe, param) {
    predictClassification(XTr, yTr, XTe, @(XTr, yTr) penLogisticRegression(XTr, yTr, 0.001, param));
};


[trRMSE, teRMSE, trLoss, teLoss] = runClassificationParam(X, y, predictParam, lambdas);

end

