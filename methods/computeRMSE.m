function [ error ] = computeRMSE( y, yHat )
%computeRMSE Computes the root mean square error
%   y       Expected result
%   yHat    Predicted result

N = length(y);
error = sqrt(sum((y - yHat).^2)./N);

end

