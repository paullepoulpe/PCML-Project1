close all
clear
clc

load('../data/SaoPaulo_regression.mat')

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
yFiltered = yNormalised;

len = length(XFiltered);
width = size(XFiltered, 2);

%% Remove correlated columns using PCA (test)
[XKept, V, VReduced] = pca(XFiltered, 1);

%% Preprocess test data
% Find cluster
for line = 1:length(yTe)
    distances = zeros(size(centers,1), 1);
    for group = 1:size(centers,1)
        distances(group) = norm(centers(group, :) - [XTe(line, :) yTe(line,:)]);
    end
    group = find(distances == min(distances));
    clusterTrue(line) = group;
end
clusterTrue = clusterTrue';

% Normalise given mean and std of the training set
XTe = (XTe - ones(size(XTe,1),1)*meanX)./(ones(size(XTe,1),1)*stdX);
yTe = (yTe - meanY)./stdY;
% Reduce the dimension given PCA of training set
XTe = (VReduced'*(V*XTe'))';

%% Train for clusters
tXTr = [ones(length(XKept), 1)  XKept];
tXTe = [ones(length(XTe), 1)  XTe];
y = cluster;

beta = logRegMultiClass( y, tXTr, 0.001, 4 );
clusterPred = predictClass(tXTe, beta, 4);

figure()
plot(clusterTrue, clusterPred,'.','markersize',10);
axis([0 5 0 5]);

%% Train
y = yFiltered;

% res = logisticRegression(y, tX, 0.01);
% res = penLogisticRegression(y, tX, 0.01,1000);
% res = leastSquaresGD(y, tX, 0.01);
res = leastSquares(y, tXTr);
% res = ridgeRegression(y, tX, 10);

yPred = tXTr * res;
yTrue = yFiltered;

figure()
plot(yTrue, yPred, '*');
hold on;
maxY = ceil(max([yPred yTrue]));
minY = floor(min([yPred yTrue]));
plot(minY:maxY, minY:maxY);