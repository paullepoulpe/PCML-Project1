function X = normalise(X)

N = size(X,1);
X = (X - ones(N,1)*mean(X))./(ones(N,1)*std(X));

end