function [ error ] = computeRMSE( y, yHat )
%computeRMSE

N = length(y);
error = sqrt(sum((y - yHat).^2)./N);

end

