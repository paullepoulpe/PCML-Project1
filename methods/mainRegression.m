close all
clear
clc

load('../data/SaoPaulo_regression.mat')

[ trRMSE, teRMSE ] = crossValidationRegression( y_train, X_train, 4 );