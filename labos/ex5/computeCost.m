function L = computeCost(y, tX, beta)

N = length(y);
e = y - tX*beta; %compute error
L = e'*e/(2*N); %compute MSE

end