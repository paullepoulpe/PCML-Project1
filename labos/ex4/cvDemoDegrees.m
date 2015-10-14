% load data
clear all;
close;
clc;

load('dataEx3.mat');

% choose degree
degree = (1:7);

for j = 1:length(degree)
    
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
            tXTr = [ones(length(yTr), 1) myPoly(XTr, degree(j))];
            tXTe = [ones(length(yTe), 1) myPoly(XTe, degree(j))];
            
            % least squares (INSERT CODE)
            [beta]  =  ridgeRegression(yTr,  tXTr,  lambda(i));
            
            % training and test RMSE(INSERT CODE)
            mseTrSub(k) = sqrt(2*computeCost(yTr,tXTr,beta));
            
            % testing RMSE using least squares
            mseTeSub(k) = sqrt(2*computeCost(yTe,tXTe,beta));
            
        end
        mseTr(i,j) = mean(mseTrSub);
        mseTe(i,j) = mean(mseTeSub);
    end
    
end

mseTrBest = mean(mseTr,1);
mseTeBest = mean(mseTe,1);

% plot

figure()
ax(1) = subplot(211);
boxplot(mseTr);
title('Train RMSE for model degree, on different lambda');
ax(2) = subplot(212);
boxplot(mseTe);
title('Test RMSE for model degree, on different lambda');

figure()
plot(degree, mseTrBest, 'bo', degree, mseTeBest, 'ro');
legend('Training best lambda for each degree','Testing best lambda for each degree');
xlabel('degree')