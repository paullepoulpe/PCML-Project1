function [L] = computeCost(y,tX,beta)
N = size(y,1);
e = y- tX*beta; %compute error
L = e'*e/(2*N); %compute MSE