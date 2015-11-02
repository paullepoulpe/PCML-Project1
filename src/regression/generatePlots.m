close all
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_regression.mat')

len = length(X_train);
dim = size(X_train, 2);

FONT_SIZE = 16;
LABEL_FONT_SIZE = 20;

setFontSize = @(gca, gcf, xl, yl) {
    set(gca, 'FontSize', FONT_SIZE); 
    set(findall(gcf,'type','text'), 'FontSize', FONT_SIZE); 
    set(xl, 'FontSize', LABEL_FONT_SIZE); set(yl, 'FontSize', LABEL_FONT_SIZE);
    };

setFontSizeNoTouchTicks = @(gca, gcf, xl, yl) {
    %set(gca, 'FontSize', FONT_SIZE); 
    %set(findall(gcf,'type','text'), 'FontSize', FONT_SIZE); 
    set(xl, 'FontSize', LABEL_FONT_SIZE); set(yl, 'FontSize', LABEL_FONT_SIZE);
    };

getShortLabels = @(nums) {
    arrayfun(@(x) sprintf('%.1E', x), nums, 'UniformOutput', false);
    };


%% Histogram of Y
figure();
histogram(y_train);
xl = xlabel('y values');
yl = ylabel('Frequency');
setFontSize(gca, gcf, xl, yl);
print -dpng 'plots/y_histogram.png';
close;

%% Dimension 55
figure();
plot(X_train(:, 55), y_train, '*');
xl = xlabel('55th dimension');
yl = ylabel('y');
setFontSize(gca, gcf, xl, yl);
print -dpng 'plots/dim55.png';
close;

%% Dimension 6
figure();
plot(X_train(:, 6), y_train, '*');
xl = xlabel('6th dimension');
yl = ylabel('y');
setFontSize(gca, gcf, xl, yl);
print -dpng 'plots/dim6.png';
close;

%% Plot trained clusters
figure();
hold on;
[cluster, predict] = trainClusters(X_train, y_train);
plot(X_train(cluster == 1, 55), X_train(cluster == 1, 6), 'r*');
plot(X_train(cluster == 2, 55), X_train(cluster == 2, 6), 'b*');
plot(X_train(cluster == 3, 55), X_train(cluster == 3, 6), '*', 'MarkerEdgeColor', [0,0.5,0]);
legend('Cluster 1', 'Cluster 2', 'Cluster 3','Location','northwest');
xl = xlabel('55th dimension');
yl = ylabel('6th dimension');
setFontSize(gca, gcf, xl, yl);
print -dpng 'plots/clusters_train.png';
close;

%% Plot test clusters phase 1

figure();
hold on;
cluster = predict(X_test);
plot(X_test(cluster == 3, 55), X_test(cluster == 3, 6), 'b*');
plot(X_test(cluster == 1, 55), X_test(cluster == 1, 6), 'r*');
plot(X_test(cluster == 2, 55), X_test(cluster == 2, 6), 'r*');
legend('Cluster 1', 'Cluster 2','Location','northwest');
xl = xlabel('55th dimension');
yl = ylabel('6th dimension');
setFontSize(gca, gcf, xl, yl);
print -dpng 'plots/clusters_phase1.png';
close;

%% Plot test clusters phase 2

figure();
hold on;
cluster = predict(X_test);
plot(X_test(cluster == 1, 55), X_test(cluster == 1, 6), '*', 'MarkerEdgeColor', [0,0,0]);
plot(X_test(cluster == 2, 55), X_test(cluster == 2, 6), '*', 'MarkerEdgeColor', [1,0,1]);
plot(X_test(cluster == 3, 55), X_test(cluster == 3, 6), 'w*');
legend('SubCluster 1', 'SubCluster 2','Location','northwest');
xl = xlabel('55th dimension');
yl = ylabel('6th dimension');
setFontSize(gca, gcf, xl, yl);
print -dpng 'plots/clusters_phase2.png';
close;

%% Plot lambda vs RMSE for ridge regression

[ trRMSE, teRMSE, lambda ] = findBestRidgeLambda(X_train, y_train);

%%
figure()
newLambda = logspace(log10(min(lambda)), log10(max(lambda)), 300);
newTrRMSE = spline(lambda, mean(trRMSE), newLambda);
newTeRMSE = spline(lambda, mean(teRMSE), newLambda);

ax(1) = subplot(211);
semilogx(newLambda, newTrRMSE, '-b');
grid on
grid minor
hold on
semilogx(lambda, mean(trRMSE), '.b', 'MarkerSize', 20);
xl = xlabel('Lambda');
yl = ylabel('Train RMSE');
ylim([450 800])
xlim([10^-3 10^7])
setFontSize(gca, gcf, xl, yl);

ax(2) = subplot(212);
semilogx(newLambda, newTeRMSE, '-r');
grid on
grid minor
hold on
semilogx(lambda, mean(teRMSE), '.r', 'MarkerSize', 20);
xl = xlabel('Lambda');
yl = ylabel('Test RMSE');
setFontSize(gca, gcf, xl, yl);

linkaxes(ax);
print -dpng 'plots/lambdaRMSE.png';
close;

%% Plot alpha vs RMSE for gradient descent

[ trRMSE, teRMSE, alpha ] = findBestGDAlpha(X_train, y_train);

%%
figure()
newAlpha = logspace(log10(min(alpha)), log10(max(alpha)), 300);
newTrRMSE = spline(alpha, mean(trRMSE), newAlpha);
newTeRMSE = spline(alpha, mean(teRMSE), newAlpha);

ax(1) = subplot(211);
semilogx(newAlpha, newTrRMSE, '-b');
grid on
grid minor
hold on
semilogx(alpha, mean(trRMSE), '.b', 'MarkerSize', 20);
xl = xlabel('Alpha');
yl = ylabel('Train RMSE');
ylim([450 800])
xlim([10^-3 0.33])
setFontSize(gca, gcf, xl, yl);

ax(2) = subplot(212);
semilogx(newAlpha, newTeRMSE, '-r');
grid on
grid minor
hold on
semilogx(alpha, mean(teRMSE), '.r', 'MarkerSize', 20);
xl = xlabel('Alpha');
yl = ylabel('Test RMSE');
setFontSize(gca, gcf, xl, yl);

linkaxes(ax);
print -dpng 'plots/alphaRMSE.png';
close;

% BoxPlot
figure();
ax(1) = subplot(211);
boxTr = boxplot(trRMSE,'plotstyle','compact','colors','b','labels', getShortLabels(alpha));
xl = xlabel('Alpha');
yl = ylabel('Train RMSE');
setFontSizeNoTouchTicks(gca, gcf, xl, yl);

ax(2) = subplot(212);
boxTe = boxplot(teRMSE,'plotstyle','compact','colors','r', 'labels', getShortLabels(alpha));
xl = xlabel('Alpha');
yl = ylabel('Test RMSE');
setFontSizeNoTouchTicks(gca, gcf, xl, yl);

linkaxes(ax);
print -dpng 'plots/alphaRMSEBox.png';
close;

%% Plot ridge vs leastSquares vs leastSquaresGD

lambda = 100;
alpha = 0.25;
groups = 3;
numIterations = 100;

ridgePredictor = @(y, tX) ridgeRegression(y, tX, lambda);
leastSquaresPredictor = @leastSquares;
leastSquaresGDPredictor = @(y, tX) leastSquaresGD(y, tX, alpha);

ridgePredict = @(Xtr, ytr, Xte){
    predictRegression(Xtr, ytr, Xte, ridgePredictor);
};
leastSquaresPredict = @(Xtr, ytr, Xte){
    predictRegression(Xtr, ytr, Xte, ridgePredictor);
};

leastSquaresGDPredict = @(Xtr, ytr, Xte){
    predictRegression(Xtr, ytr, Xte, ridgePredictor);
};


for i = 1:numIterations 
    fprintf('Start of iteration %d\n', i);
    [trError, teError] = crossValidation(X_train, y_train, groups, ridgePredict, false);
    ridgeTrRMSE(i) = mean(trError.RMSE);
    ridgeTeRMSE(i) = mean(teError.RMSE);
    
    [trError, teError] = crossValidation(X_train, y_train, groups, leastSquaresPredict, false);
    leastSquaresTrRMSE(i) = mean(trError.RMSE);
    leastSquaresTeRMSE(i) = mean(teError.RMSE);
    
    [trError, teError] = crossValidation(X_train, y_train, groups, leastSquaresGDPredict, false);
    leastSquaresGDTrRMSE(i) = mean(trError.RMSE);
    eastSquaresGDTeRMSE(i) = mean(teError.RMSE);
end

%%
figure()
ax(1) = subplot(121);
boxplot([leastSquaresTrRMSE' leastSquaresGDTrRMSE' ridgeTrRMSE'],'labels',{'least-squares','least-squares GD','ridge regression'});
ylabel('Train RMSE')
ax(2) = subplot(122);
boxplot([leastSquaresTeRMSE' eastSquaresGDTeRMSE' ridgeTeRMSE'],'labels',{'least-squares','least-squares GD','ridge regression'});
ylabel('Test RMSE')




