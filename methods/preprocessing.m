close all
clear
clc

load('../data/SaoPaulo_regression.mat')

%% Visualise the data 
figure()
histogram(y_train);

figure()
for i = 1:size(X_train,2)
    ax(i) = subplot(7,10,i);
    plot(X_train(:,i),y_train,'*');
end
linkaxes(ax);

%% Separe X_train in train and test to make cross-validation
[ XTr, XTe, yTr, yTe ] = divideDataSet( X_train, y_train, 4, 2 );

%% Clustering of the data with kMeans
% Need to do that with non-normalised data !
X = XTr;
y = yTr;
[cluster, centers] = kmeans([X y], 4);

figure()
plot(X(cluster==1,1),y(cluster==1,1),'*');
hold on
plot(X(cluster==2,1),y(cluster==2,1),'*');
plot(X(cluster==3,1),y(cluster==3,1),'*');
plot(X(cluster==4,1),y(cluster==4,1),'*');
plot(XTe(:,1),yTe(:,1),'*');
plot(centers(:,1), centers(:,end),'.r', 'markersize',30);
legend('cluster 1','cluster 2','cluster 3', 'cluster 4','test data','centers');
%% Normalise the data (train)

[XNormalised, meanX, stdX] = normalise(XTr);
[yNormalised, meanY, stdY] = normalise(yTr);

%% Remove outliers (train)

%[XFiltered, yFiltered] = removeOutlierLines(XNormalised, yNormalised, 3, 2);
XFiltered = fixOutliers(XNormalised, 3);
% XFiltered = XNormalised;
yFiltered = yNormalised;

len = length(XFiltered);
width = size(XFiltered, 2);

%% Remove correlated columns using PCA (test)
[XKept, V, VReduced] = pca(XFiltered, 1);

%% Preprocess test data
% Find cluster
clusterTrue = findClosestGroup([XTe yTe], centers);

% Normalise given mean and std of the training set
XTeNormalised = (XTe - ones(size(XTe,1),1)*meanX)./(ones(size(XTe,1),1)*stdX);
yTeNormalised = (yTe - meanY)./stdY;
% Reduce the dimension given PCA of training set
XTeNormalised = (VReduced'*(V*XTeNormalised'))';

%% Train for clusters
%% With logistic regression for multiclass 
tXTr = [ones(length(XKept), 1)  XKept];
tXTe = [ones(length(XTeNormalised), 1)  XTeNormalised];
y = cluster;

betaC = logRegMultiClass( y, tXTr, 0.001, 4 );
clusterPred = predictClass(tXTe, betaC, 4);

%% Train for clusters
%% With an approximation of y and minimum distance to centers
y = yFiltered;
% beta = logisticRegression(y, tX, 0.01);
% beta = penLogisticRegression(y, tX, 0.01,1000);
% beta = leastSquaresGD(y, tX, 0.01);
betaCl = leastSquares(y, tXTr);
% beta = ridgeRegression(y, tX, 10);

yPred = (tXTe * betaCl)*stdY + meanY;
clusterPred = findClosestGroup([XTe yPred], centers);

figure()
plot(clusterTrue, clusterPred,'.','markersize',10);
axis([0 5 0 5]);

%% Train
y = yFiltered;

for k = 1:4
    % beta = logisticRegression(y, tX, 0.01);
    % beta = penLogisticRegression(y, tX, 0.01,1000);
    % beta = leastSquaresGD(y, tX, 0.01);
    beta(:,k) = leastSquares(y(cluster == k), tXTr(cluster == k,:));
    % beta = ridgeRegression(y, tX, 10);
    
end
yPred = zeros(size(yTe));
for k = 1:4
    indices = find(clusterPred == k);
    for i = 1:length(indices)
        yPred(indices(i),:) = tXTe(i,:) * beta(:,k);
    end
end
yPred = yPred * stdY + meanY;
yTrue = yTe;

figure()
plot(yTrue, yPred, '*');
hold on;
maxY = ceil(max([yPred yTrue]));
minY = floor(min([yPred yTrue]));
plot(minY:maxY, minY:maxY);