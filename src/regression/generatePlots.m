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
    set(xl, 'FontSize', LABEL_FONT_SIZE); set(yl, 'FontSize', LABEL_FONT_SIZE)
    };


%% Histogram of Y
figure();
histogram(y_train);
xl = xlabel('y values');
yl = ylabel('Frequency');
setFontSize(gca, gcf, xl, yl);
print -dpdf 'plots/y_histogram.pdf';
close;

%% Dimension 55
figure();
plot(X_train(:, 55), y_train, '*');
xl = xlabel('55th dimension');
yl = ylabel('y');
setFontSize(gca, gcf, xl, yl);
print -dpdf 'plots/dim55.pdf';
close;

%% Dimension 6
figure();
plot(X_train(:, 6), y_train, '*');
xl = xlabel('6th dimension');
yl = ylabel('y');
setFontSize(gca, gcf, xl, yl);
print -dpdf 'plots/dim6.pdf';
close;

%% Plot trained clusters
figure();
hold on;
[cluster, predict] = trainClusters(X_train, y_train);
plot(X_train(cluster == 1, 55), X_train(cluster == 1, 6), 'r*');
plot(X_train(cluster == 2, 55), X_train(cluster == 2, 6), 'b*');
plot(X_train(cluster == 3, 55), X_train(cluster == 3, 6), '*', 'MarkerEdgeColor', [0,0.5,0]);
legend('Cluster 1', 'Cluster 2', 'Cluster 3');
xl = xlabel('55th dimension');
yl = ylabel('6th dimension');
setFontSize(gca, gcf, xl, yl);
print -dpdf 'plots/clusters_train.pdf';
close;

%% Plot test clusters phase 1

figure();
hold on;
cluster = predict(X_test);
plot(X_test(cluster == 3, 55), X_test(cluster == 3, 6), 'b*');
plot(X_test(cluster == 1, 55), X_test(cluster == 1, 6), 'r*');
plot(X_test(cluster == 2, 55), X_test(cluster == 2, 6), 'r*');
legend('Cluster 1', 'Cluster 2');
xl = xlabel('55th dimension');
yl = ylabel('6th dimension');
setFontSize(gca, gcf, xl, yl);
print -dpdf 'plots/clusters_phase1.pdf';
close;

%% Plot test clusters phase 2

figure();
hold on;
cluster = predict(X_test);
plot(X_test(cluster == 1, 55), X_test(cluster == 1, 6), '*', 'MarkerEdgeColor', [0,0,0]);
plot(X_test(cluster == 2, 55), X_test(cluster == 2, 6), '*', 'MarkerEdgeColor', [1,0,1]);
plot(X_test(cluster == 3, 55), X_test(cluster == 3, 6), 'w*');
legend('SubCluster 1', 'SubCluster 2');
xl = xlabel('55th dimension');
yl = ylabel('6th dimension');
setFontSize(gca, gcf, xl, yl);
print -dpdf 'plots/clusters_phase2.pdf';
close;

%% Plot lambda vs RMSE for ridge regression

[ trRMSE, teRMSE, lambda ] = crossValidationParam(X_train, y_train, 3, @predictRegression, 3);

%%
figure()
newLambda = logspace(-3, 7, 300);
newTrRMSE = spline(lambda, mean(trRMSE), newLambda);
newTeRMSE = spline(lambda, mean(teRMSE), newLambda);

ax(1) = subplot(211);
boxTr = semilogx(newLambda, newTrRMSE, '-b');
grid on
grid minor
hold on
semilogx(lambda, mean(trRMSE), '*b');
xl = xlabel('Train RMSE');
yl = ylabel('lambda');
ylim([450 800])
xlim([10^-3 10^7])
setFontSize(gca, gcf, xl, yl);

ax(2) = subplot(212);
boxTe = semilogx(newLambda, newTeRMSE, '-r');
grid on
grid minor
hold on
semilogx(lambda, mean(teRMSE), '*r');
xl = xlabel('Test RMSE');
yl = ylabel('lambda');
setFontSize(gca, gcf, xl, yl);

linkaxes(ax);



