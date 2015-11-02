function [ error ] = computeLogLoss( y, pHat )
%computeLogLoss

N = length(y);

y(y<0) = 0;

error = 0;
for i = 1:N
    if(pHat(i) < 1)
    error = error + y(i) .* log(pHat(i)) + (1 - y(i)) .* log(1 - pHat(i)); 
    else 
        error = error + y(i) .* log(pHat(i));
    end
end
   
error = -error./N;

end
