function [ finalClusters, predict ] = trainClusters( X, y )
%trainCluster Trains the data to find which cluster a certain point belongs to
%
% clusters      clusters for the training data
% predict       function to predict clusters for the test data

% We have three clusters, and two dimensions that correlate highly with y, 
% so we use them to separate the clusters
[I, ~] = correlation(X, y);

% Use most correlated dimensions to separate first two clusters
[clusters, centers, sumd] = kmeansStable([X(:, I(1)) y], 2);

% Use second most correlated dimension to separate sparse cluster in 2
clusterSizes = [ sum(clusters == 1); sum(clusters == 2) ];
clusterSpread = sumd ./ clusterSizes;
sparseCluster = find(clusterSpread == max(clusterSpread));
subX = X(clusters == sparseCluster, :);
suby = y(clusters == sparseCluster, :);
[subclusters, subcenters, ~] = kmeansStable([subX(: , I(2)) suby], 2);

% Relabel clusters for global labeling
finalClusters = clusters;
finalClusters(clusters ~= sparseCluster) = 3;
finalClusters(clusters == sparseCluster) = subclusters;

predict = @predictCluster;


function finalClustersPred = predictCluster(XTe)
% Predict the clusters for test data
    
clustersPred = findClosestGroup(XTe(:, I(1)), centers(:, 1));
subClustersPred = findClosestGroup(XTe(clustersPred == sparseCluster, I(2)), subcenters(:, 1));

finalClustersPred = clustersPred;
finalClustersPred(clustersPred ~= sparseCluster) = 3;
finalClustersPred(clustersPred == sparseCluster) = subClustersPred;

end

end

