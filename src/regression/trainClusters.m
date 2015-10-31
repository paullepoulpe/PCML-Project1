function [ finalClusters, predict ] = trainClusters( X, y, dim1, dim2)
%trainCluster Trains the data to find which cluster a certain point belongs to
%
% clusters      clusters for the training data
% predict       function to predict clusters for the test data

[clusters, centers, ~] = kmeansStable([X(:, dim1) y], 2);
[subclusters, subcenters, ~] = kmeansStable([X(clusters == 1, dim2) y(clusters == 1)], 2);

finalClusters = clusters;
finalClusters(clusters == 2) = 3;
finalClusters(clusters == 1) = subclusters;

predict = @predictCluster;


function finalClustersPred = predictCluster(XTe)

clustersPred = findClosestGroup(XTe(:, dim1), centers(:, 1));
subClustersPred = findClosestGroup(XTe(clustersPred == 1, dim2), subcenters(:, 1));

finalClustersPred = clustersPred;
finalClustersPred(clustersPred == 2) = 3;
finalClustersPred(clustersPred == 1) = subClustersPred;

end

end

