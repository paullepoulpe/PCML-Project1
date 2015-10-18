function [L] = computeCostMae(y,tX,beta)
N = size(y,1);
e = y- tX*beta; %compute error
L = sum(abs(e))/(2*N); %compute MAE