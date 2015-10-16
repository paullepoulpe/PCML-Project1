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
height = (height - mean(height))./std(height);
weight = (weight - mean(weight))./std(weight);
% Form (y,tX) to get regression data in matrix form
Xnormalised = [height weight];
y = gender;
N = length(y);
tX = [ones(N,1) Xnormalised];


% algorithm parametes
maxIters = 1000;
alpha = 0.001;
converged = 0;

% initialize
beta = [0; 0; 0];

% iterate
fprintf('Starting iterations, press Ctrl+c to break\n');
fprintf('L  beta0 betaH betaW\n');
for k = 1:maxIters
    % INSERT YOUR FUNCTION FOR COMPUTING GRADIENT
    g = computeGradient(y,tX,beta);
    
    % INSERT YOUR FUNCTION FOR COMPUTING COST FUNCTION
    L = computeCostLogisticReg(y, tX, beta);
    
    % INSERT GRADIENT DESCENT UPDATE TO FIND BETA
    beta = beta - alpha.*g;
    
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
    
%     % Overlay on the contour plot
%     % For this to work you first have to run grid Search
%     subplot(121);
%     plot(beta(1), beta(2), 'o', 'color', 0.7*[1 1 1], 'markersize', 12);
%     pause(.5) % wait half a second
%     
%     % visualize function f on the data
%     subplot(122);
%     x = [1.2:.01:2]; % height from 1m to 2m
%     x_normalized = (x - meanX)./stdX;
%     f = beta(1) + beta(2).*x_normalized;
%     plot(height, weight,'.');
%     hold on;
%     plot(x,f,'r-');
%     hx = xlabel('x');
%     hy = ylabel('y');
%     hold off;
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
contourf(hx, wx, pred, 1);
% plot indiviual data points
hold on
myBlue = [0.06 0.06 1];
myRed = [1 0.06 0.06];
plot(Xnormalised(males,1), Xnormalised(males,2),'x','color',myRed,'linewidth', ...
    2, 'markerfacecolor', myRed);
hold on
plot(Xnormalised(females,1), Xnormalised(females,2),'o','color', ...
    myBlue,'linewidth', 2, 'markerfacecolor', myBlue);
xlabel('height');
ylabel('weight');
xlim([min(h) max(h)]);
ylim([min(w) max(w)]);
grid on;
