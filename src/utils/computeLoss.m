function [ error ] = computeLoss( y, yHat )
%computeLoss

N = length(y);

yDiff = y - yHat;

error = sum(yDiff ~= 0)./N;

end

