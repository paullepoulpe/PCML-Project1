close all
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_classification.mat')

len = length(X_train);
dim = size(X_train, 2);

FONT_SIZE = 16;
LABEL_FONT_SIZE = 20;

setFontSize = @(gca, gcf, xl, yl) {
    set(gca, 'FontSize', FONT_SIZE); 
    set(findall(gcf,'type','text'), 'FontSize', FONT_SIZE); 
    set(xl, 'FontSize', LABEL_FONT_SIZE); set(yl, 'FontSize', LABEL_FONT_SIZE)
    };


%% Histogram of dimension 35
figure();
X35 = X_train(:, 35);

[~, xbins] = hist(X35, length(unique(floor(X35))));
type1 = X35(X_train(:, 20) == 1);
type2 = X35(X_train(:, 20) == 2);
bars1 = hist(type1, xbins);
bars2 = hist(type2, xbins);
b = bar(xbins, [bars1 ; bars2]', 'stacked');
b(1).FaceColor = 'b';
b(2).FaceColor = 'r';
l = cell(1,2);
l{1}='20th dimension == 1'; 
l{2}='20th dimension == 2';    
legend(b,l);
xl = xlabel('35th dimension''s values');
yl = ylabel('Frequency');
setFontSize(gca, gcf, xl, yl);

print -dpng 'plots/dim35.png';
close;

%% Histogram of dimension 3
figure();
X23 = X_train(:, 23);

[~, xbins] = hist(X23, length(unique(round(X23, -2))));
type1 = X23(X_train(:, 20) == 1);
type2 = X23(X_train(:, 20) == 2);
bars1 = hist(type1, xbins);
bars2 = hist(type2, xbins);
b = bar(xbins, [bars1 ; bars2]', 'stacked');
b(1).FaceColor = 'b';
b(2).FaceColor = 'r';
l = cell(1,2);
l{1}='20th dimension == 1'; 
l{2}='20th dimension == 2';    
legend(b,l);
xl = xlabel('3rd dimension''s values');
yl = ylabel('Frequency');
setFontSize(gca, gcf, xl, yl);

print -dpng 'plots/dim23.png';
close;

% Dimension 55
% figure();
% plot(X_train(:, 55), y_train, '*');
% xl = xlabel('55th dimension');
% yl = ylabel('y');
% setFontSize(gca, gcf, xl, yl);
% print -dpdf 'dim55.pdf';
% close;
% 
% Dimension 6
% figure();
% plot(X_train(:, 6), y_train, '*');
% xl = xlabel('6th dimension');
% yl = ylabel('y');
% setFontSize(gca, gcf, xl, yl);
% print -dpdf 'dim6.pdf';
% close;
% 
% Plot trained clusters
% figure();
% hold on;
% [cluster, predict] = trainClusters(X_train, y_train);
% plot(X_train(cluster == 1, 55), X_train(cluster == 1, 6), 'r*');
% plot(X_train(cluster == 2, 55), X_train(cluster == 2, 6), 'b*');
% plot(X_train(cluster == 3, 55), X_train(cluster == 3, 6), '*', 'MarkerEdgeColor', [0,0.5,0]);
% legend('Cluster 1', 'Cluster 2', 'Cluster 3');
% xl = xlabel('55th dimension');
% yl = ylabel('6th dimension');
% setFontSize(gca, gcf, xl, yl);
% print -dpdf 'clusters_train.pdf';
% close;
% 
% Plot test clusters phase 1
% 
% figure();
% hold on;
% cluster = predict(X_test);
% plot(X_test(cluster == 3, 55), X_test(cluster == 3, 6), 'b*');
% plot(X_test(cluster == 1, 55), X_test(cluster == 1, 6), 'r*');
% plot(X_test(cluster == 2, 55), X_test(cluster == 2, 6), 'r*');
% legend('Cluster 1', 'Cluster 2');
% xl = xlabel('55th dimension');
% yl = ylabel('6th dimension');
% setFontSize(gca, gcf, xl, yl);
% print -dpdf 'clusters_phase1.pdf';
% close;
% 
% Plot test clusters phase 2
% 
% figure();
% hold on;
% cluster = predict(X_test);
% plot(X_test(cluster == 1, 55), X_test(cluster == 1, 6), '*', 'MarkerEdgeColor', [0,0,0]);
% plot(X_test(cluster == 2, 55), X_test(cluster == 2, 6), '*', 'MarkerEdgeColor', [1,0,1]);
% plot(X_test(cluster == 3, 55), X_test(cluster == 3, 6), 'w*');
% legend('SubCluster 1', 'SubCluster 2');
% xl = xlabel('55th dimension');
% yl = ylabel('6th dimension');
% setFontSize(gca, gcf, xl, yl);
% print -dpdf 'clusters_phase2.pdf';
% close;




