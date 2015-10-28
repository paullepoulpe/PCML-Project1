function [ res ] = cardinalities( X )
%CARDINALITIES Counts the cardinality of each columns
%   Returns for each column, the count of unique elements it has

dims = size(X, 2);

res = zeros(1, dims);
for i =  1:dims
    res(i) = length(unique(X(:, i)));
end


end

