clear
close all
clc

for s = 1:50 % # of seeds
    setSeed(s);
    % generate data
    N = 50;
    X = linspace(0.1,2*pi,N)';
    y = sin(X(:)) + 0.3*randn(N,1);
    % randomly permute data
    idx = randperm(N);
    y = y(idx);
    X = X(idx);
    % split data
    [XTr, yTr, XTe, yTe] = split(y,X,0.8); % PROPORTION !!!
    
    % for degree k
    degrees = [1:7];
    for k = 1:length(degrees)
        % get beta using least squares
        tXTr = [ones(length(yTr), 1) myPoly(XTr, k)];
        tXTe = [ones(length(yTe), 1) myPoly(XTe, k)];
        beta = leastSquares(yTr, tXTr);
        
        % compute train and test RMSE
        rmseTr(s,k) = sqrt(2*computeCost(yTr,tXTr,beta));
        % 4.4 Visualizing bias-variance decomposition 3
        rmseTe(s,k) = sqrt(2*computeCost(yTe,tXTe,beta));
    end
end
% compute expected train and test error
rmseTr_mean = mean(rmseTr);
rmseTe_mean = mean(rmseTe);
% plot
figure()
plot(degrees, rmseTe,'r-','color',[1 0.7 0.7]);
hold on
plot(degrees, rmseTr,'b-','color',[0.7 0.7 1]);
plot(degrees, rmseTe_mean,'r-','linewidth', 3);
hold on
plot(degrees, rmseTr_mean,'b-','linewidth', 3);
xlabel('degree');
ylabel('error');

figure()
boxplot(rmseTe, 'boxstyle', 'filled');