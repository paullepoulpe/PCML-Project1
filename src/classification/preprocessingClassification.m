close all
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_classification.mat')


%% Separe X_train in train and test to make cross-validation
[ XTr, XTe, yTr, yTe ] = divideDataSet( X_train, y_train, 4, 2 );

% cards = cardinalities( XTr );
% XTr = XTr(:,cards < size(XTr,1));
% XTe = XTe(:,cards < size(XTr,1));

%% Separe data set in function of the 20th dimension of X
cluster = XTr(:,20);
clusterPred = XTe(:,20);

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
    
    % Normalise the data
    [XTrNormalised, meanX(cl,:), stdX(cl,:)] = normalise(X);
    yTrNormalised = y;
    yTrNormalised(yTrNormalised < 0) = 0;

    % Remove outliers
    [XTrFiltered, yTrFiltered] = removeOutlierLines(XTrNormalised, yTrNormalised, 3, 1);
%     XFiltered = fixOutliers(XTrNormalised, 3);

    % Remove correlated columns using PCA 
%     [XTrKept, V{cl,1}, VReduced{cl,1}] = pca(XTrFiltered, 1);
    XTrKept = XTrFiltered;
    
    % Compute tX and final y
    tXTr{cl,1} = [ones(length(XTrKept), 1)  XTrKept];
    yTrF{cl,1} = yTrFiltered;
    
    % beta{cl,1} = logisticRegression(yTrF{cl,1}, tXTr{cl,1}, 0.001);
    beta{cl,1} = penLogisticRegression(yTrF{cl,1}, tXTr{cl,1}, 0.001, 20);

end

%% Test
yPred = zeros(size(XTe,1),1);
for cl = 1:length(unique(clusterPred))
    % Take the data from one cluster
    X = XTe(finalClusterPred == cl,:);
    
    % Normalise given mean and std of the training set
    XTeNormalised = (X - ones(size(X,1),1)*meanX(cl,:))./(ones(size(X,1),1)*stdX(cl,:));
    
    % Reduce the dimension given PCA of training set
%     XTeNormalised = (VReduced{cl,1}'*(V{cl,1}*XTeNormalised'))';
    
    % Compute tX
    tXTe = [ones(length(XTeNormalised), 1)  XTeNormalised];
    
    % Prediction
    yPred(clusterPred == cl,:) = sigma(tXTe * beta{cl,1});

end

yPred(yPred >= 0.5) = 1;
yPred(yPred < 0.5) = -1;
yTrue = yTe;

figure()
plot(yTrue, yPred, '*');
