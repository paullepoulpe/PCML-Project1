close all
clear
clc

load('SaoPaulo_regression.mat')

%% Preprocessing the data

XNormalised = normalise(X_train);
yNormalised = normalise(y_train);

%% Remove outliers

for i = 1:length(XNormalised)
    co
