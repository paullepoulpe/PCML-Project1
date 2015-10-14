% load data
clear all;
close;
clc;

load('dataEx3.mat');

% choose degree
degree = 7;

% split data in K fold (we will only create indices)
setSeed(1);
K = 4;
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% lambda values (INSERT CODE)
lambda  =  logspace(-2,2,100);

% K-fold cross validation
for i = 1:length(lambda)
	for k = 1:K
		% get k'th subgroup in test, others in train
		idxTe = idxCV(k,:);
		idxTr = idxCV([1:k-1 k+1:end],:);
		idxTr = idxTr(:);
		yTe = y(idxTe);
		XTe = X(idxTe,:);
		yTr = y(idxTr);
		XTr = X(idxTr,:);
		% form tX (INSERT CODE)
        tXTr = [ones(length(yTr), 1) myPoly(XTr, degree)];
        tXTe = [ones(length(yTe), 1) myPoly(XTe, degree)];

		% least squares (INSERT CODE)
        [beta]  =  ridgeRegression(yTr,  tXTr,  lambda(i));

		% training and test RMSE(INSERT CODE)
		mseTrSub(k) = sqrt(2*computeCost(yTr,tXTr,beta)); 

        % testing RMSE using least squares
		mseTeSub(k) = sqrt(2*computeCost(yTe,tXTe,beta));

	end
	mseTr(i) = mean(mseTrSub);
	mseTe(i) = mean(mseTeSub);
end

% plot

figure()
semilogx(lambda,mseTr,'bo',lambda, mseTe,'ro');
legend('Train RMSE','Test RMSE');