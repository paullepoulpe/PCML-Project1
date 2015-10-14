% load data
clear all;
load('dataEx3.mat');

for degree = [3 7 12]
fprintf('degree %d\n', degree);
for proportion = [0.9, 0.5, 0.1]
	% get train and test data
	[XTr, yTr, XTe, yTe] = split(y,X,proportion);
	% form tX
	tXTr = [ones(length(yTr), 1) myPoly(XTr, degree)];
	tXTe = [ones(length(yTe), 1) myPoly(XTe, degree)];

	% least squares
	% INSERT YOUR LEASTSQUARES FUNCTION HERE
	beta = leastSquares(yTr, tXTr); %zeros(size(tXTr,2),1);

	% train and test MSE
	rmseTr = sqrt(2*computeCost(yTr,tXTr,beta)); 
	rmseTe = sqrt(2*computeCost(yTe,tXTe,beta)); 

	% print 
	fprintf('Proportion %.2f: Train RMSE :%0.4f Test RMSE :%0.4f\n', proportion, rmseTr, rmseTe);
end
fprintf('press any key to continue\n');
pause
end


