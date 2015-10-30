close all
clear
clc

load('../data/SaoPaulo_regression.mat')

% [ trRMSE, teRMSE ] = crossValidationRegression( y_train, X_train, 3 );
predict = @(Xtr, Ytr, Xte) predictRegression(Xtr, Ytr, Xte, 10^5);

[ trRMSE, teRMSE ] = crossValidation( X_train, y_train, 3, predict);