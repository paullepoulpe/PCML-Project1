clc
clear
close all

load('dataEx3.mat');

% get train and test data
[XTr, yTr, XTe, yTe] = split(y,X,0.5);
degree = 7;

% form tX
tXTr = [ones(length(yTr), 1) myPoly(XTr, degree)];
tXTe = [ones(length(yTe), 1) myPoly(XTe, degree)];

%  given  the  split  (yTr,  XTr)  and  (yTe,  XTe)
vals  =  logspace(-2,2,100);
for  i  =  1:length(vals)
lambda  =  vals(i);
%  ridge  regression
[beta]  =  ridgeRegression(yTr,  tXTr,  lambda);

%  compute  training  error
errTr(i)  =  computeCost(yTr,  tXTr,  beta);
%  compute  test  error
errTe(i)  =  computeCost(yTe,  tXTe,  beta);
end
[errStar,  lambdaStar]  =  min(errTe);

figure()
semilogx(vals,errTr,'b',vals,errTe,'r')