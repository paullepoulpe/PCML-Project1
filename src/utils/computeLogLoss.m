function [ error ] = computeLogLoss( y, pHat )
%computeLogLoss

N = length(y);

y(y<0) = 0;

error = - sum(y .* log(pHat) + (ones(size(y))-y) .* log(ones(size(pHat))-pHat))./N;

end
