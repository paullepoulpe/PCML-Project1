function [ trRMSE, teRMSE, param ] = crossValidationParam( X, y, groups, predictParam, nbIter)
%crossValidationRegression

% param = logspace(-6, 10, 60); % For ridge regression
param = logspace(-3, 3, 10); % For penalised logistic regression
lenParam = length(param);

trRMSE = zeros(1, lenParam);
teRMSE = zeros(1, lenParam);

for paramIdx = 1:lenParam
    
    lambda = param(paramIdx);
    predict = @(Xtr, ytr, Xte) predictParam(Xtr, ytr, Xte, lambda);
    
    trMeanError = zeros(1, nbIter);
    teMeanError = zeros(1, nbIter);
    
    for iter = 1:nbIter
        [trError, teError] = crossValidation(X, y, groups, predict);
        trMeanError(iter) = mean(trError);
        teMeanError(iter) = mean(teError);
    end
    
    trMeanError = trMeanError( abs(normalise(trMeanError)) < 2 );
    teMeanError = teMeanError( abs(normalise(teMeanError)) < 2 );

    trRMSE(paramIdx) = mean(trMeanError);
    teRMSE(paramIdx) = mean(teMeanError);
    
    fprintf('Finished %d out of %d\n', paramIdx, lenParam);
    
end

end

