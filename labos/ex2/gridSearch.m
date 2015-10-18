% INSERT CODE WHERE INDICATED
% Written by Emtiyaz, EPFL for PCML 2014, 2015
% all rights reserved
  clear all

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

  % generate a grid of values for beta0 and beta1
  beta0 = [-100:5:200];
  beta1 = [-150:5:150];
  L = zeros(length(beta0), length(beta1));
  % INSERT THE CODE FOR GRID SEARCH HERE
  tic;
  for i=1:length(beta0)
      for j=1:length(beta1)
          L(i,j) = computeCostMae(y,tX,[beta0(i);beta1(j)]);
      end
  end
  timeGridSearch = toc

  % compute minimum value of L and also beta0_star and beta1_star
  [val, idx] = min(L(:));
  [i_min,j_min] = ind2sub(size(L), idx);
  beta0_star = beta0(i_min);
  beta1_star = beta1(j_min);
  L_star = L(i_min, j_min);

  % print the output
  fprintf('L*, beta0*, beta1*\n');
  fprintf('%.2f, %.2f %.2f\n', L_star, beta0_star, beta1_star);

  % contour plot
  figure(1)
  clf;
  subplot(121);
  contourf(beta0, beta1, L', 20); colorbar;
  hx = xlabel('\beta_0');
  hy = ylabel('\beta_1');
  hold on
  plot(beta0_star, beta1_star, 'h',...% put a marker at the minimum
          'markersize', 12, 'markerfacecolor',[1 1 1]...
          ,'markeredgecolor',[1 1 1], 'linewidth',1);
  set(gca, 'fontsize', 14);

  % visualize function f on the data
  subplot(122);
  x = [1.2:.01:2]; % height from 1m to 2m
  x_normalized = (x - meanX)./stdX;
  f = beta0_star + beta1_star*x_normalized;
  plot(height, weight,'.');
  hold on;
  plot(x,f,'r-','linewidth', 2);
  hx = xlabel('x');
  hy = ylabel('y');
  grid on;
  hold off;

