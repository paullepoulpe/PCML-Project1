close all
clear
clc

load('../data/SaoPaulo_classification.mat')

%% Separe X_train in train and test to make cross-validation
[ XTr, XTe, yTr, yTe ] = divideDataSet( X_train, y_train, 4, 2 );

% cards = cardinalities( XTr );
% XTr = XTr(:,cards < size(XTr,1));
% XTe = XTe(:,cards < size(XTr,1));
%% Preprocessing the data
[XNormalised, meanX, stdX] = normalise(XTr);
yNormalised = yTr;
yNormalised(yNormalised < 0) = 0;

%% Remove outliers
%[XFiltered, yFiltered] = removeOutlierLines(XNormalised, yNormalised, 3, 2); 
XFiltered = fixOutliers(XNormalised, 3);
% XFiltered = XNormalised;
yFiltered = yNormalised;

len = length(XFiltered);
width = size(XFiltered, 2);

%% Remove correlated columns using PCA
% XKept = pca(XFiltered, 1);
XKept = XFiltered;

%% Preprocess test data
% Normalise given mean and std of the training set
XTeNormalised = (XTe - ones(size(XTe,1),1)*meanX)./(ones(size(XTe,1),1)*stdX);
yTeNormalised = yTe;
% Reduce the dimension given PCA of training set
% XTeNormalised = (VReduced'*(V*XTeNormalised'))';

%% Train
tXTr = [ones(size(XKept,1), 1)  XTr];
tXTe = [ones(length(XTeNormalised), 1)  XTeNormalised];
y = yTr;

% beta = logisticRegression(y, tXTr, 0.001);
beta = penLogisticRegression(y, tXTr, 0.001,20);

yPred = sigma(tXTe * beta);
yPred(yPred >= 0.5) = 1;
yPred(yPred < 0.5) = 0;
yTrue = yTe;

yPred(yPred == 0) = -1; 

figure()
plot(yTrue, yPred, '*');
