function [ groups ] = findClosestGroup( X, centers )
%findClosestGroup Find group of the data by computing the minimum distance
% to the centers of the groups

groups = dsearchn(centers, X);

end
