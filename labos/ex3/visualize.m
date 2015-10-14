% load data
clear all
load('dataEx3.mat');

% basis function models
vals = [1 3 7 12];
for k = 1:length(vals)
	% form tX for polynomials
	degree = vals(k);
	Xpoly = myPoly(X,degree);
	tX = [ones(length(y),1) Xpoly];

	% INSERT YOUR LEASTSQUARES FUNCTION HERE
	beta = leastSquares(y, tX); %zeros(size(tX,2),1); 

	% compute RMSE
	rmsePoly = sqrt(2*computeCost(y,tX,beta));
	fprintf('MSE polynomial (degree %d) %.4f\n', degree, rmsePoly);

	% plot fit
	nr = 2; nc = 2;
	subplot(nr,nc,k);
	plot(X, y, 'ob', 'markersize', 4); % plot data
	hold on;
	xvals = [min(X)-0.1:.1:max(X)+.1];
	tX = [ones(length(xvals),1) myPoly(xvals(:), degree)];
	f = tX*beta;
	plot(xvals, f,'r-','linewidth',2);
	xlim([min(xvals), max(xvals)]);
	hx = xlabel('x');
	hy = ylabel('y');
	ht = title(sprintf('Polynomial degree %d',degree));
	set([hx, hy, ht], 'fontsize', 14);
end
