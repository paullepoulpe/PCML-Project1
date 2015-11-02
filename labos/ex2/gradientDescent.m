% INSERT CODE WHERE INDICATED
% Written by Emtiyaz, EPFL for PCML 2014, 2015
% all rights reserved
  clear all

  % run grid search first to get the baseline
  gridSearch;

  % Load data and comvert it to the metrics system
  load('height_weight_gender.mat');
  height = height * 0.025;
  weight = weight * 0.454;
  
  % For outlier studying : subsample, get only 1 for every 50 points
  height = height(1:50:end);
  weight = weight(1:50:end);
  % simulate outliers
  % weight in pounds instead of kilos
  height(end+1) = 1.1;
  weight(end+1) = 51.5 / 0.454;
  height(end+1) = 1.2;
  weight(end+1) = 55.3 / 0.454;
  
  % normalize features (store the mean and variance)
  x = height;
  meanX = mean(x);
  x = x - meanX;
  stdX = std(x);
  x = x./stdX;

  % Form (y,tX) to get regression data in matrix form
  y = weight;
  N = length(y);
  tX = [ones(N,1) x(:)];


  % algorithm parametes
  maxIters = 1000;
  alpha = 0.1;
  converged = 0;

  % initialize
  beta = [0; 0];

  % iterate
  fprintf('Starting iterations, press Ctrl+c to break\n');
  fprintf('L  beta0 beta1\n');
  for k = 1:maxIters
    % INSERT YOUR FUNCTION FOR COMPUTING GRADIENT
    g = computeGradient(y,tX,beta);

    % INSERT YOUR FUNCTION FOR COMPUTING COST FUNCTION
    L = computeCost(y,tX,beta);

    % INSERT GRADIENT DESCENT UPDATE TO FIND BETA
    beta = beta-alpha.*g;

    % INSERT CODE FOR CONVERGENCE
    if k>1
        if abs(beta(1)-beta_all(1,k-1))< 0.001
            if abs(beta(2)-beta_all(2,k-1))< 0.001
                converged = k
                break;
            end
        end
    end
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;

    % print
    fprintf('%.2f  %.2f %.2f\n', L, beta(1), beta(2));

    % Overlay on the contour plot
    % For this to work you first have to run grid Search
    subplot(121);
    plot(beta(1), beta(2), 'o', 'color', 0.7*[1 1 1], 'markersize', 12);
    pause(.5) % wait half a second

    % visualize function f on the data
    subplot(122);
    x = [1.2:.01:2]; % height from 1m to 2m
    x_normalized = (x - meanX)./stdX;
    f = beta(1) + beta(2).*x_normalized;
    plot(height, weight,'.');
    hold on;
    plot(x,f,'r-');
    hx = xlabel('x');
    hy = ylabel('y');
    hold off;
  end

