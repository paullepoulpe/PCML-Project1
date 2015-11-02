%% Plot all the dimensions of X_train to visually inspect
close all
clear
clc

load('../../data/SaoPaulo_classification.mat');

%X_train = fixOutliers(X_train, 1);
% keep = zeros(1, length(X_train));
% for i = 1:length(X_train) 
%     line = X_train(i, :);
%     if(line(20) == 1)
%         keep(i) = 0;
%     else 
%         keep(i) = 1;
%     end
% end
% X_train = X_train(keep == 1, :);
% y_train = y_train(keep == 1, :);
% 
% [X_train, y_train] = removeOutlierLines(X_train, y_train, 3, 1);

len = length(X_train);
dim = size(X_train, 2);


for d = 1:dim
   figure();
%    hold on;
%    plot(X_train(y_train == -1, 30), X_train(y_train == -1, d), 'r*');
%    plot(X_train(y_train == 1, 30), X_train(y_train == 1, d), 'b*');
   histogram(X_train(:, d))
end