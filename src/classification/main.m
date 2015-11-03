close all
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_classification.mat')

predictor = @(y, tX) penLogisticRegression(y, tX, 0.001, 0.01);
predict = @(Xtr, ytr, Xte) predictClassification(Xtr, ytr, Xte, predictor);

[yTe, pTe] = predict(X_train, y_train, X_test);
csvwrite('results/predictions_classification.csv', pTe);


nbIteration = 50;

lossFunction = zeros(1, nbIteration);
for i = 1:nbIteration
    home;
    fprintf('Iteration %d out of %d\n', i, nbIteration);
    [~, teError] = crossValidation(X_train, y_train, 2, predict, true);
    lossFunction(i) = mean(teError.Loss);
end

file = fopen('results/test_errors_classification.csv', 'w');
fprintf(file, 'loss,%1.3f', mean(lossFunction));
fclose(file);