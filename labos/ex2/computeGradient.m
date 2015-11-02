function [g] = computeGradient(y,tX,beta)
N = size(y,1);
g = -1/N * tX'*(y-tX*beta);
