function [ X, meanX, stdX ] = normalise(X)

N = size(X,1);
meanX = mean(X);
stdX = std(X);
X = (X - ones(N,1)*meanX)./(ones(N,1)*stdX);

end