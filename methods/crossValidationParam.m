function [ trRMSE, teRMSE, param ] = crossValidationParam( X, y, groups, predictParam)
%crossValidationRegression

param = logspace(-5, 8, 20);
lenParam = length(param);

trRMSE = zeros(1, lenParam);
teRMSE = zeros(1, lenParam);

for paramIdx = 1:lenParam
    
    lambda = param(paramIdx);
    
    predict = @(Xtr, ytr, Xte) predictParam(Xtr, ytr, Xte, lambda);
    
    [trError, teError] = crossValidation(X, y, groups, predict);
    
    trRMSE(paramIdx) = mean(trError);
    teRMSE(paramIdx) = mean(teError);
    
end

end

