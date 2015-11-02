function [ trRMSE, teRMSE ] = runRegressionParam( X, y, predictParam, params )

groups = 3; % Number of groups
nbIter = 3; % Number of times to run cross validation

lenParam = length(params); 

trRMSE = zeros(nbIter, lenParam);
teRMSE = zeros(nbIter, lenParam);

for paramIdx = 1:lenParam
    
    lambda = params(paramIdx);
    predict = @(Xtr, ytr, Xte) predictParam(Xtr, ytr, Xte, lambda);
    
    trMeanRMSE = zeros(1, nbIter);
    teMeanRMSE = zeros(1, nbIter);
    
    for iter = 1:nbIter
        [trError, teError] = crossValidation(X, y, groups, predict, false);
        trMeanRMSE(iter) = mean(trError.RMSE);
        teMeanRMSE(iter) = mean(teError.RMSE);
    end
    
    trRMSE(:,paramIdx) = trMeanRMSE;
    teRMSE(:,paramIdx) = teMeanRMSE;
    
    fprintf('Finished %d out of %d\n', paramIdx, lenParam);
    
end

end

