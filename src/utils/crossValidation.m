function [ trError, teError ] = crossValidation( X, y, groups, predict, doComputeLoss)
%crossValidationRegression Returns RMSE for train and test for this prediciton function
% X                 Data input
% y                 Output
% groups            Number of groups to form
% predict           The prediciton function
% doComputeLoss     Set to true to compute loss

trRMSE = zeros(1, groups);
teRMSE = zeros(1, groups);

if doComputeLoss
    trLoss = zeros(1, groups);
    teLoss = zeros(1, groups);
    trLogLoss = zeros(1, groups);
    teLogLoss = zeros(1, groups);
end

for groupNumber = 1:groups
    %% Separe X and y in train and test parts
    [ XTr, XTe, yTr, yTe ] = divideDataSet( X, y, groups, groupNumber );
    
    %% Make prediction for test data
    yTrPred = predict( XTr, yTr, XTr );
    yTePred = predict( XTr, yTr, XTe );
    
    % TODO HACK FIXME: for some reason we get cells instea of arrays
    if iscell(yTrPred) 
        yTrPred = yTrPred{1};
    end
    if iscell(yTePred) 
        yTePred = yTePred{1};
    end
    
    
    % RMSE
    trRMSE(groupNumber) = computeRMSE(yTr, yTrPred);
    teRMSE(groupNumber) = computeRMSE(yTe, yTePred);
    
    % Loss function
    if doComputeLoss
        trLoss(groupNumber) = computeLoss(yTr, yTrPred);
        teLoss(groupNumber) = computeLoss(yTe, yTePred);
        
        trLogLoss(groupNumber) = computeLogLoss(yTr, yTrPred);
        teLogLoss(groupNumber) = computeLogLoss(yTe, yTePred);
    end
end

trError.RMSE = trRMSE;
teError.RMSE = teRMSE;

if doComputeLoss
    trError.Loss = trLoss;
    teError.Loss = teLoss;
    
    trError.LogLoss = trLogLoss;
    teError.LogLoss = teLogLoss;
end

end

