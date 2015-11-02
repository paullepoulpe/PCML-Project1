clear
close all
clc

% run grid search first to get the baseline
% gridSearch;

% Load data and comvert it to the metrics system
load('height_weight_gender.mat');
height = height * 0.025;
weight = weight * 0.454;
X = [height weight];
% normalize features (store the mean and variance)
meanHeight = mean(height);
heightNormalised = height - meanHeight;
stdHeight = std(height);
heightNormalised = heightNormalised./stdHeight;
meanWeight = mean(weight);
weightNormalised = weight - meanWeight;
stdWeight = std(weight);
weightNormalised = weightNormalised./stdWeight;

% Form (y,tX) to get regression data in matrix form
Xnormalised = [heightNormalised weightNormalised];
y = gender;
N = length(y);
tX = [ones(N,1) Xnormalised];


% algorithm parametes
maxIters = 1000;
alpha = 0.1;
lambda = 100;
converged = 0;

% initialize
beta = [0; 0; 0];

% iterate
fprintf('Starting iterations, press Ctrl+c to break\n');
fprintf('L  beta0 betaH betaW\n');
for k = 1:maxIters
    
    [L, g, H] = logisticRegLoss(beta, y, tX);
%     [L, g, H] = logisticPenRegLoss(beta, y, tX,lambda);
    
    % INSERT NEWTON METHOD UPDATE TO FIND BETA
    beta = beta - alpha.*(H^-1)*g;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
    
    % print
    fprintf('%.2f  %.2f %.2f %.2f\n', L, beta(1), beta(2), beta(3));
    
    % INSERT CODE FOR CONVERGENCE
    if k>1
        if abs(L-L_all(k-1))<0.1
            converged = 1
            break;
        end
    end
end

beta = beta_all(:,k);

% create a 2?D meshgrid of values of heights and weights
h = [min(Xnormalised(:,1)):.01:max(Xnormalised(:,1))];
w = [min(Xnormalised(:,2)):.01:max(Xnormalised(:,2))];
[hx, wx] = meshgrid(h,w);
% predict for each pair, i.e. create tX for each [hx,wx]
% and then predict the value. After that you should
% reshape `pred` so that you can use `contourf`.
% For this you need to understand how `meshgrid` works.

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

for i=1:size(hx,1)
    for j=1:size(hx,2)
        pred(i,j) = computeSigma([1 hx(i,j) wx(i,j)]*beta);
        if pred(i,j)>0.5
            pred(i,j)=1;
        else
            pred(i,j)=0;
        end
    end
end

% plot the decision surface
figure()
contourf(hx*stdHeight+meanHeight, wx*stdWeight+meanWeight, pred, 1);
colormap(jet)
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
grid on;
