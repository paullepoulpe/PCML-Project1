function [ groups, centers ] = kmeans( k, X )
%KMEANS Summary of this function goes here
%   Detailed explanation goes here

dim = size(X, 2);
len = length(X);

% Pre sort
norms = zeros(len, 1);
for line = 1:len
    norms(line) = norm(X(line));
end
[~, I] = sort(norms);

presorted = X(I, :);


% Iniitialize centers
centers = zeros(k, dim);
for cluster = 1:k
   sizeChunk = floor(len/k);
   start = (cluster - 1) * sizeChunk + 1;
   ennd = min(cluster * sizeChunk + 1, len);
   centers(cluster, :) = mean(presorted(start:ennd ,:));
end

groups = zeros(len, 1);
old_centers = zeros(k, dim);
while centers ~= old_centers

    % Recompute cluster membership
    for line = 1:len
       distances = zeros(k, 1);
       for cluster = 1:k
           distances(cluster) = norm(centers(cluster, :) - X(line, :));
       end
       group = find(distances == min(distances));
       groups(line) = group;
    end
   
    
    % Recompute cluster centers
    old_centers = centers;
    for cluster = 1:k
        centers(cluster, :) = mean(X(groups == cluster, :));
    end  
    
end

end

