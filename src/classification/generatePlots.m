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

setFontSizeNoTouchTicks = @(gca, gcf, xl, yl) {
    %set(gca, 'FontSize', FONT_SIZE); 
    %set(findall(gcf,'type','text'), 'FontSize', FONT_SIZE); 
    set(xl, 'FontSize', LABEL_FONT_SIZE); set(yl, 'FontSize', LABEL_FONT_SIZE);
    };

getShortLabels = @(nums) {
    arrayfun(@(x) sprintf('%.1E', x), nums, 'UniformOutput', false);
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

%% BoxPlots of lambda vs Loss

[ trRMSE, teRMSE, trLoss, teLoss, lambda ] = findBestPenLambda(X_train, y_train);

%%
figure()
boxplot(trLoss,'plotstyle','compact','colors','b','labels', getShortLabels(lambda));
hold on
boxplot(teLoss,'plotstyle','compact','colors','r', 'labels', getShortLabels(lambda));
xl = xlabel('Lambda');
yl = ylabel('Loss');

setFontSizeNoTouchTicks(gca, gcf, xl, yl);

ylim([0 0.15])

print -dpng 'plots/lambdaLossBox.png';
close;

