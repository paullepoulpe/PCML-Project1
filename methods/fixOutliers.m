function [ XFixed ] = fixOutliers( X, limitDeviation )
%fixOutliers Gets the outliers back to a few standard deviations away
%
% limitDeviation    Number of std deviations away that makes a point an
%                   outlier

stdX = std(X);
meanX = mean(X);
len = length(X);

XNormalised = normalise(X);

XNormalised(XNormalised > limitDeviation) = limitDeviation;
XNormalised(XNormalised < -limitDeviation) = -limitDeviation;

XFixed = (XNormalised .* (ones(len,1)*stdX)) +  ones(len,1)*meanX;

end

