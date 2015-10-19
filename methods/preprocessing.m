close all
clear
clc

load('./data/SaoPaulo_regression.mat')

%% Preprocessing the data

XNormalised = normalise(X_train);
yNormalised = normalise(y_train);

%% Remove outliers

%[XFiltered, yFiltered] = removeOutlierLines(XNormalised, yNormalised, 3, 2); 
XFiltered = fixOutliers(XNormalised, 3);
yFiltered = yNormalised;

len = length(XFiltered);
width = size(XFiltered, 2);

%% Remove close columns
deviations = zeros(width, width);
for i = 1:width
   for j = 1:width
       col1 = XFiltered(:, i);
       col2 = XFiltered(:, j);
       deviations(i, j) = std(max(col1 ./ col2, col2 ./col1));
   end
end

% Almost colinear columns
[row, col] = find(deviations > 0 & deviations < 3);
toKeep = setdiff(1:width, col);

XKept = XFiltered; % XFiltered(:, toKeep);

%% Train
trainSize = 2500;
testSize = length(XKept) - trainSize;

train = XKept(1:trainSize, :);
tx = [ones(length(train), 1)  train];
y = yFiltered(1:trainSize, :);

% res = logisticRegression(y, tx, 0.01);
res = leastSquaresGD(y, tx, 0.01);
% res = leastSquares(y, tx);
% res = ridgeRegression(y, tx, 10);

y_pred = [ones(testSize , 1) XKept((trainSize + 1):end,:)] * res;
y_true = yFiltered((trainSize + 1):end, :);

plot(y_true, y_pred, '*');
hold on;
max = ceil(max([y_pred y_true]));
min = floor(min([y_pred y_true]));
plot(min:max, min:max);