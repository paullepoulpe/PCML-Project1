close all
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_regression.mat')


predictor = @(y, tX) ridgeRegression(y, tX, 100);
predict = @(Xtr, ytr, Xte) predictRegression(Xtr, ytr, Xte, predictor);

yTe = predict(X_train, y_train, X_test);
csvwrite('results/predictions_regression.csv', yTe);


nbIteration = 100;

RMSE = zeros(1, nbIteration);
for i = 1:nbIteration
    home;
    fprintf('Iteration %d out of %d\n', i, nbIteration);
    [~, teError] = crossValidation(X_train, y_train, 3, predict, false);
    RMSE(i) = mean(teError.RMSE);
end

file = fopen('results/test_errors_regression.csv', 'w');
fprintf(file, 'rmse,%d', round(mean(RMSE)));
fclose(file);

