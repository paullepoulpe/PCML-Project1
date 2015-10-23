function [ groups ] = findClosestGroup( X, centers )
%findClosestGroup Find group of the data by computing the minimum distance
% to the centers of the groups
for line = 1:length(X)
    distances = zeros(size(centers,1), 1);
    for group = 1:size(centers,1)
        distances(group) = norm(centers(group, :) - X(line, :));
    end
    group = find(distances == min(distances));
    groups(line) = group;
end
groups = groups';
end
