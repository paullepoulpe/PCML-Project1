function [ yPred ] = predictRegression( XTr, yTr, XTe, param )
%predictRegression Prediction function for the regression

%% Find cardinalities of data
cards = cardinalities(XTr);
clusteredDim = [55, 6];

%% Remove dimensions with discrete arguments
XTr = XTr(:,cards == size(XTr,1));
XTe = XTe(:,cards == size(XTr,1));
clusteredDim = [46, 6];

%% Clustering of the data with kmeansStable
% Need to do that with non-normalised data !

[clusters, predictClusters] = trainClusters(XTr, yTr, clusteredDim(1), clusteredDim(2));
numClusters = length(unique(clusters));
%% Predict test clusters 

clustersPred = predictClusters(XTe);

%figure()
%plot(XTe(clustersPred == 1, clusteredDim(1)),XTe(clustersPred == 1, clusteredDim(2)),'r*');
%hold on
%plot(XTe(clustersPred == 2, clusteredDim(1)),XTe(clustersPred == 2, clusteredDim(2)),'b*');
%plot(XTe(clustersPred == 3, clusteredDim(1)),XTe(clustersPred == 3, clusteredDim(2)),'g*');

%% Train regression model for each cluster
V = cell(numClusters, 1);
VReduced = cell(numClusters, 1);
tXTr = cell(numClusters, 1);
yTrF = cell(numClusters, 1);
beta = cell(numClusters, 1);

for cl = 1:numClusters
    % Take the data for one cluster
    X = XTr(clusters == cl,:);
    y = yTr(clusters == cl,:);
    
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
    
%     beta{cl,1} = leastSquaresGD(yTrF{cl,1}, tXTr{cl,1}, param);
%     beta{cl,1} = leastSquares(yTrF{cl,1}, tXTr{cl,1});
    beta{cl,1} = ridgeRegression(yTrF{cl,1}, tXTr{cl,1}, param);
    
%     figure()
%     plot(XTrKept(:,1),yTrF{cl,1},'*');
end

%% Test 
yPred = zeros(size(XTe,1),1);
for cl = 1:numClusters
    % Take the data from one cluster
    X = XTe(clustersPred == cl,:);
    %y = yTr(finalClusterPred == cl,:);
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
    yPred(clustersPred == cl,:) = tXTe * beta{cl,1};
    yPred(clustersPred == cl,:) = yPred(clustersPred == cl,:) * stdY(cl,:) + meanY(cl,:);
end

end

