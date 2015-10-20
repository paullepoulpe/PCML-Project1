close all
clear
clc

load('../data/SaoPaulo_regression.mat')

%% Preprocessing the data

XNormalised = normalise(X_train);
yNormalised = normalise(y_train);

%% Remove outliers

%[XFiltered, yFiltered] = removeOutlierLines(XNormalised, yNormalised, 3, 2); 
XFiltered = fixOutliers(XNormalised, 3);
yFiltered = yNormalised;

len = length(XFiltered);
width = size(XFiltered, 2);

%% Remove correlated columns using PCA
XKept = pca(XFiltered, 1);

%% Train
trainSize = 2500;
testSize = length(XKept) - trainSize;

train = XKept(1:trainSize, :);
tX = [ones(length(train), 1)  train];
y = yFiltered(1:trainSize, :);

% res = logisticRegression(y, tX, 0.01);
% res = penLogisticRegression(y, tX, 0.01,1000);
% res = leastSquaresGD(y, tX, 0.01);
res = leastSquares(y, tX);
% res = ridgeRegression(y, tX, 10);

y_pred = [ones(testSize , 1) XKept((trainSize + 1):end,:)] * res;
y_true = yFiltered((trainSize + 1):end, :);

plot(y_true, y_pred, '*');
hold on;
max = ceil(max([y_pred y_true]));
min = floor(min([y_pred y_true]));
plot(min:max, min:max);