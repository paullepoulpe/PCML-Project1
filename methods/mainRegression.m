close all
clear
clc

load('../data/SaoPaulo_regression.mat')

% predict = @(Xtr, Ytr, Xte) predictRegression(Xtr, Ytr, Xte, 10^5);
% [ trRMSE, teRMSE ] = crossValidation(y_train, X_train, 3, predict);

[ trRMSE, teRMSE, lambda ] = crossValidationParam(X_train, y_train, 3, @predictRegression);