close all
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_classification.mat')

% predict = @(Xtr, Ytr, Xte) predictRegression(Xtr, Ytr, Xte, 10^5);
% [ trRMSE, teRMSE ] = crossValidation(y_train, X_train, 3, predict);

[ trRMSE, teRMSE, lambda ] = crossValidationParam(X_train, y_train, 3, @predictClassification, 4);

% semilogx(lambda, teRMSE, 'r*-')
% hold on
% semilogx(lambda, trRMSE, 'b*-')

%%
figure()
% subplot(211)
boxTr = boxplot(trRMSE,'plotstyle','compact','colors','b','labels',lambda);
% subplot(212)
hold on
boxTe = boxplot(teRMSE,'plotstyle','compact','colors','r');
ylim([0.05 0.12])