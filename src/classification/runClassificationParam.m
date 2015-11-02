function [ trRMSE, teRMSE, trLoss, teLoss ] = runClassificationParam( X, y, predictParam, params )

groups = 2; % Number of groups
nbIter = 3; % Number of times to run cross validation

lenParam = length(params); 

trRMSE = zeros(nbIter, lenParam);
teRMSE = zeros(nbIter, lenParam);

trLoss = zeros(nbIter, lenParam);
teLoss = zeros(nbIter, lenParam);

for paramIdx = 1:lenParam
    
    lambda = params(paramIdx);
    predict = @(Xtr, ytr, Xte) predictParam(Xtr, ytr, Xte, lambda);
    
    trMeanRMSE = zeros(1, nbIter);
    teMeanRMSE = zeros(1, nbIter);
    
    trMeanLoss = zeros(1, nbIter);
    teMeanLoss = zeros(1, nbIter);
    
    for iter = 1:nbIter
        [trError, teError] = crossValidation(X, y, groups, predict, true);
        
        trMeanRMSE(iter) = mean(trError.RMSE);
        teMeanRMSE(iter) = mean(teError.RMSE);
        
        trMeanLoss(iter) = mean(trError.Loss);
        teMeanLoss(iter) = mean(teError.Loss);
    end
    
    trRMSE(:,paramIdx) = trMeanRMSE;
    teRMSE(:,paramIdx) = teMeanRMSE;
    
    trLoss(:,paramIdx) = trMeanLoss;
    teLoss(:,paramIdx) = teMeanLoss;
    
    fprintf('Finished %d out of %d\n', paramIdx, lenParam);
    
end

end

