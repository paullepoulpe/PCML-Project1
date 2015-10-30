function [ yPred ] = predict( XTr, yTr, XTe )
%predict

%% Find cardinalities of data
cards = cardinalities(XTr);
clusteredDim = [55, 6];

%% Remove dimensions with discrete arguments
XTr = XTr(:,cards == size(XTr,1));
XTe = XTe(:,cards == size(XTr,1));
clusteredDim = [46, 6];

%% Clustering of the data with kmeansStable
% Need to do that with non-normalised data !
X = XTr;
y = yTr;
[cluster, centers, ~] = kmeansStable([X(:,clusteredDim(1)) y], 2);
[subcluster, subcenters, ~] = kmeansStable([X(cluster==1,clusteredDim(2)) y(cluster==1)], 2);

finalCluster = cluster;
finalCluster(cluster == 2) = 3;
finalCluster(cluster==1) = subcluster;
finalCenters = [subcenters(1,:);centers(2,:);subcenters(2,:)];

%% Normalise the data (train)

[XTrNormalised, meanX, stdX] = normalise(XTr);
[yTrNormalised, meanY, stdY] = normalise(yTr);

%% Remove outliers (train)

XTrFiltered = fixOutliers(XTrNormalised, 3);
yTrFiltered = yTrNormalised;
% [XTrFiltered, yFiltered] = removeOutlierLines(XTrNormalised, [yTrNormalised finalCluster], 3, 1);
% finalCluster = yFiltered(:,2);
% yTrFiltered = yFiltered(:,1);
%% Remove correlated columns using PCA (test)
[XTrKept, V, VReduced] = pca(XTrFiltered, 1);

%% Preprocess test data
% Normalise given mean and std of the training set
XTeNormalised = (XTe - ones(size(XTe,1),1)*meanX)./(ones(size(XTe,1),1)*stdX);
% Reduce the dimension given PCA of training set
XTeNormalised = (VReduced'*(V*XTeNormalised'))';

%% Compute tX for train and test
tXTr = [ones(length(XTrKept), 1)  XTrKept];
tXTe = [ones(length(XTeNormalised), 1)  XTeNormalised];

%% Train for clusters using minimum distance to the center in 55th dimension
clusterPred = findClosestGroup([XTe(:,clusteredDim(1))], centers(:,1));
finalClusterPred = clusterPred;
finalClusterPred(clusterPred == 2) = 3;

subClusterPred = findClosestGroup([XTe(finalClusterPred~=3,clusteredDim(2))], finalCenters(1:2,1));
finalClusterPred(finalClusterPred~=3) = subClusterPred;

figure()
plot(XTe(finalClusterPred==1,clusteredDim(1)),XTe(finalClusterPred==1,clusteredDim(2)),'*');
hold on
plot(XTe(finalClusterPred==2,clusteredDim(1)),XTe(finalClusterPred==2,clusteredDim(2)),'*');
plot(XTe(finalClusterPred==3,clusteredDim(1)),XTe(finalClusterPred==3,clusteredDim(2)),'*');

%% Train
y = yTrFiltered;

for k = 1:3
    %     beta(:,k) = leastSquaresGD(y(finalCluster == k), tXTr(finalCluster == k,:), 0.01);
        beta(:,k) = leastSquares(y(finalCluster == k), tXTr(finalCluster == k,:));
%     beta(:,k) = ridgeRegression(y(finalCluster == k), tXTr(finalCluster == k,:), 0.1);
    
end

yPred = zeros(size(XTe,1),1);
for k = 1:3
    indices = find(finalClusterPred == k);
    for i = 1:length(indices)
        yPred(indices(i),:) = tXTe(i,:) * beta(:,k);
    end
end

yPred = yPred * stdY + meanY;

end

