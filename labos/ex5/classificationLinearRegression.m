close all
clear
clc

% Load data
load('height_weight_gender.mat');
height = height * 0.025;
weight = weight * 0.454;
y = gender;
X = [height(:)  weight(:)];
% randomly permute data
N = length(y);
idx = randperm(N);
y = y(idx);
X = X(idx,:);
% subsample
y = y(1:200);
X = X(1:200,:);

% Plot the data
males = [];
females = [];
for i=1:length(y)
    if y(i)==true
        males = [males, i];
    else
        females = [females, i];
    end
end

figure()
myBlue = [0.06 0.06 1];
myRed = [1 0.06 0.06];
plot(X(males,1), X(males,2),'xr','color',myRed,'linewidth', ...
    2, 'markerfacecolor', myRed);
hold on
plot(X(females,1), X(females,2),'or','color', ...
    myBlue,'linewidth', 2, 'markerfacecolor', myBlue);
xlabel('height');
ylabel('weight');


% create a 2?D meshgrid of values of heights and weights
h = [min(X(:,1)):.01:max(X(:,1))];
w = [min(X(:,2)):1:max(X(:,2))];
[hx, wx] = meshgrid(h,w);
% predict for each pair, i.e. create tX for each [hx,wx]
% and then predict the value. After that you should
% reshape `pred` so that you can use `contourf`.
% For this you need to understand how `meshgrid` works.

degree = 1;
tXTr = [ones(length(y), 1) X];
beta = leastSquares(y,tXTr);

for i=1:size(hx,1)
    for j=1:size(hx,2)
        pred(i,j) = beta(1)+beta(2)*hx(i,j)+beta(3)*wx(i,j);
    end
end

% plot the decision surface
figure()
contourf(hx, wx, pred, 1);
% plot indiviual data points
hold on
myBlue = [0.06 0.06 1];
myRed = [1 0.06 0.06];
plot(X(males,1), X(males,2),'x','color',myRed,'linewidth', ...
    2, 'markerfacecolor', myRed);
hold on
plot(X(females,1), X(females,2),'o','color', ...
    myBlue,'linewidth', 2, 'markerfacecolor', myBlue);
xlabel('height');
ylabel('weight');
xlim([min(h) max(h)]);
ylim([min(w) max(w)]);
grid on;