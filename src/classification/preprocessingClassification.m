close all
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_classification.mat')


%% Separe X_train in train and test to make cross-validation
[ XTr, XTe, yTr, yTe ] = divideDataSet( X_train, y_train, 4, 2 );

%% Separe data set in function of the 20th dimension of X
cluster = XTr(:,20);
clusterPred = XTe(:,20);

%%
cards = cardinalities( XTr );
XTr = XTr(:,cards > 2);
XTe = XTe(:,cards > 2);

%% Train regression model for each cluster
sizeCluster = length(unique(cluster));
V = cell(sizeCluster,1);
VReduced = cell(sizeCluster,1);
tXTr = cell(sizeCluster,1);
yTrF = cell(sizeCluster,1);
beta = cell(sizeCluster,1);

for cl = 1:sizeCluster
    % Take the data for one cluster
    X = XTr(cluster == cl,:);
    y = yTr(cluster == cl,:);
    
    cardsX(cl,:) = cardinalities( X );
    disX = X(:,cardsX(cl,:) ~= size(X,1));
    contX = X(:,cardsX(cl,:) == size(X,1));
    
    % Normalise the data
    [XTrNormalised, meanX(cl,:), stdX(cl,:)] = normalise(contX);
    
    y(y < 0) = 0;

    % Remove outliers
    [XTrFiltered, yTrFiltered, linesKept] = removeOutlierLines(XTrNormalised, y, 3, 1);

    disX = disX(linesKept == 1,:);
    XTrKept = [XTrFiltered, disX];
    
    % Compute tX and final y
    tXTr{cl,1} = [ones(length(XTrKept), 1)  XTrKept];
    yTrF{cl,1} = yTrFiltered;
    
%     beta{cl,1} = logisticRegression(yTrF{cl,1}, tXTr{cl,1}, 0.001);
    beta{cl,1} = penLogisticRegression(yTrF{cl,1}, tXTr{cl,1}, 0.001, 0.01);

end

%% Test
pPred = zeros(size(XTe,1),1);
yPred = zeros(size(XTe,1),1);
for cl = 1:length(unique(clusterPred))
    % Take the data from one cluster
    X = XTe(clusterPred == cl,:);
    disX = X(:,cardsX(cl,:) < 10);
    contX = X(:,cardsX(cl,:) >= 10);
    
    % Normalise given mean and std of the training set
    XTeNormalised = (contX - ones(size(contX,1),1)*meanX(cl,:))./(ones(size(contX,1),1)*stdX(cl,:));
    
    XTeKept = [XTeNormalised, disX];
    
    % Compute tX
    tXTe = [ones(length(XTeNormalised), 1)  XTeKept];
    
    % Prediction
    pPred(clusterPred == cl,:) = sigma(tXTe * beta{cl,1});

end

yPred(pPred >= 0.5) = 1;
yPred(pPred < 0.5) = -1;
yTrue = yTe;

find(yPred-yTrue ~=0)

rmse = computeRMSE(yTrue, yPred);
loss = computeLoss(yTrue, yPred);
% logLoss = computeLogLoss( yTrue, pPred );
