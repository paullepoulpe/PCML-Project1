%% Plot all the dimensions of X_train to visually inspect
clear
clc
close all

load('../../data/SaoPaulo_regression.mat');
len = length(X_train);
dim = size(X_train, 2);

for d = 1:dim
   figure();
   hold on;
   plot(X_train(:, d), y_train, '*');
end