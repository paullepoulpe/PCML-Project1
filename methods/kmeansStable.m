function [ groups, centers, sumd ] = kmeansStable( X, k )
%KMEANS Clustering X data in k clusters
% We need this function because the native MATLAB kmeans algorithm uses 
% random heuristics to initialise the centers

dim = size(X, 2);
len = length(X);

% Pre sort
norms = zeros(len, 1);
for line = 1:len
    norms(line) = norm(X(line));
end
[~, I] = sort(norms);

presorted = X(I, :);


% Inititialize centers
centers = zeros(k, dim);
for cluster = 1:k
   sizeChunk = floor(len/k);
   start = (cluster - 1) * sizeChunk + 1;
   ennd = min(cluster * sizeChunk + 1, len);
   centers(cluster, :) = mean(presorted(start:ennd, :));
end

[groups, centers, sumd] = kmeans(X, k, 'Start', centers);

end

