function [ I, corrs ] = correlation( X, y )
%CORRELATION Returns the indices with highest correlation between input and output 
%   corrs returns the correlation

dim = size(X, 2);

corrs = zeros(1, dim);
for d = 1:dim
   column = X(:, d);
   corrs(d) = norm(corr(column, y));
end

[corrs, I] = sort(corrs, 2, 'DESCEND');

end

