close all
clear
clc

load('../data/SaoPaulo_classification.mat')


%% Separe X_train in train and test to make cross-validation
[ XTr, XTe, yTr, yTe ] = divideDataSet( X_train, yTrFiltered, 4, 2 );

% cards = cardinalities( XTr );
% XTr = XTr(:,cards < size(XTr,1));
% XTe = XTe(:,cards < size(XTr,1));

%% Separe data set in function of the 20th dimension of X
cluster = XTr(:,20);
clusterPred = XTe(:,20);

%% Train regression model for each cluster
V = cell(size(finalCenters,1),1);
VReduced = cell(size(finalCenters,1),1);
tXTr = cell(size(finalCenters,1),1);
yTrF = cell(size(finalCenters,1),1);
beta = cell(size(finalCenters,1),1);

for cl = 1:length(unique(cluster))
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
    [XTrKept, V{cl,1}, VReduced{cl,1}] = pca(XTrFiltered, 1);
    XTrKept = XTrFiltered;
    
    % Compute tX and final y
    tXTr{cl,1} = [ones(length(XTrKept), 1)  XTrKept];
    yTrF{cl,1} = yTrFiltered;
    
    % beta{cl,1} = logisticRegression(yTrF{cl,1}, tXTr{cl,1}, 0.001);
    beta{cl,1} = penLogisticRegression(yTrF{cl,1}, tXTr{cl,1}, 0.001, 20);

end

len = length(XFiltered);
width = size(XFiltered, 2);

%% Test
yPred = zeros(size(XTe,1),1);
for cl = 1:length(unique(clusterPred))
    
% Normalise given mean and std of the training set
XTeNormalised = (XTe - ones(size(XTe,1),1)*meanX)./(ones(size(XTe,1),1)*stdX);
yTeNormalised = yTe;
% Reduce the dimension given PCA of training set
% XTeNormalised = (VReduced'*(V*XTeNormalised'))';

%% Train
tXTr = [ones(size(XKept,1), 1)  XTr];
tXTe = [ones(length(XTeNormalised), 1)  XTeNormalised];
y = yTr;



yPred = sigma(tXTe * beta);
yPred(yPred >= 0.5) = 1;
yPred(yPred < 0.5) = 0;
yTrue = yTe;

yPred(yPred == 0) = -1; 

figure()
plot(yTrue, yPred, '*');
