close all
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_classification.mat')

% predict = @(Xtr, Ytr, Xte) predictRegression(Xtr, Ytr, Xte, 10^5);
% [ trRMSE, teRMSE ] = crossValidation(y_train, X_train, 3, predict);

