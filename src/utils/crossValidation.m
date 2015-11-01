function [ trError, teError ] = crossValidation( X, y, groups, predict)
%crossValidationRegression Returns RMSE for train and test for this prediciton function
% X         Data input
% y         Output
% groups    Number of groups to form
% predict   The prediciton function

trError = zeros(1, groups);
teError = zeros(1, groups);

for groupNumber = 1:groups
    %% Separe X and y in train and test parts
    [ XTr, XTe, yTr, yTe ] = divideDataSet( X, y, groups, groupNumber );
    
    %% Make prediction for test data
    yTrPred = predict( XTr, yTr, XTr );
    yTePred = predict( XTr, yTr, XTe );
    
    % RMSE
%     trError(groupNumber) = computeRMSE(yTr, yTrPred);
%     teError(groupNumber) = computeRMSE(yTe, yTePred);
    % Loss function
    trError(groupNumber) = computeLoss(yTr, yTrPred);
    teError(groupNumber) = computeLoss(yTe, yTePred);
    
end

end

