function [ yPred ] = predictRegression( XTr, yTr, XTe, param )
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

%% Predict test clusters using 55 and 6th dimensions
clusterPred = findClosestGroup([XTe(:,clusteredDim(1))], centers(:,1));
finalClusterPred = clusterPred;
finalClusterPred(clusterPred == 2) = 3;

subClusterPred = findClosestGroup([XTe(finalClusterPred~=3,clusteredDim(2))], finalCenters(1:2,1));
finalClusterPred(finalClusterPred~=3) = subClusterPred;

% figure()
% plot(XTe(finalClusterPred==1,clusteredDim(1)),XTe(finalClusterPred==1,clusteredDim(2)),'*');
% hold on
% plot(XTe(finalClusterPred==2,clusteredDim(1)),XTe(finalClusterPred==2,clusteredDim(2)),'*');
% plot(XTe(finalClusterPred==3,clusteredDim(1)),XTe(finalClusterPred==3,clusteredDim(2)),'*');

%% Train regression model for each cluster
V = cell(size(finalCenters,1),1);
VReduced = cell(size(finalCenters,1),1);
tXTr = cell(size(finalCenters,1),1);
yTrF = cell(size(finalCenters,1),1);

for cl = 1:size(finalCenters,1)
    % Take the data for one cluster
    X = XTr(finalCluster == cl,:);
    y = yTr(finalCluster == cl,:);
    
    % Normalise the data
    [XTrNormalised, meanX(cl,:), stdX(cl,:)] = normalise(X);
    [yTrNormalised, meanY(cl,:), stdY(cl,:)] = normalise(y);
    
    %% Remove outliers
    XTrFiltered = fixOutliers(XTrNormalised, 3);
%     XTrFiltered = XTrNormalised;
    yTrFiltered = yTrNormalised;
    % [XTrFiltered, yFiltered] = removeOutlierLines(XTrNormalised, [yTrNormalised finalCluster], 3, 1);
    % finalCluster = yFiltered(:,2);
    % yTrFiltered = yFiltered(:,1);
    
    % Remove correlated columns using PCA 
    [XTrKept, V{cl,1}, VReduced{cl,1}] = pca(XTrFiltered, 1);
    
    % Compute tX and final y
    tXTr{cl,1} = [ones(length(XTrKept), 1)  XTrKept];
    yTrF{cl,1} = yTrFiltered;
    
%     beta(:,cl) = leastSquaresGD(yTrF{cl,1}, tXTr{cl,1}, param);
%     beta(:,cl) = leastSquares(yTrF{cl,1}, tXTr{cl,1});
    beta(:,cl) = ridgeRegression(yTrF{cl,1}, tXTr{cl,1}, param);
end

%% Test 
yPred = zeros(size(XTe,1),1);
for cl = 1:size(finalCenters,1)
    % Take the data from one cluster
    X = XTr(finalClusterPred == cl,:);
    y = yTr(finalClusterPred == cl,:);
    
    % Normalise given mean and std of the training set
    XTeNormalised = (X - ones(size(X,1),1)*meanX(cl,:))./(ones(size(X,1),1)*stdX(cl,:));
    
    % Fix outliers
    limitDeviation = 3;
    XTeNormalised(XTeNormalised > limitDeviation) = limitDeviation;
    XTeNormalised(XTeNormalised < -limitDeviation) = -limitDeviation;
    
    % Reduce the dimension given PCA of training set
    XTeNormalised = (VReduced{cl,1}'*(V{cl,1}*XTeNormalised'))';
    
    % Compute tX
    tXTe = [ones(length(XTeNormalised), 1)  XTeNormalised];
    
    % Compute the prediction
    yPred(finalClusterPred == cl,:) = tXTe * beta(:,cl);
    yPred(finalClusterPred == cl,:) = yPred(finalClusterPred == cl,:) * stdY(cl,:) + meanY(cl,:);
end

end

