function [ trRMSE, teRMSE ] = crossValidationRegression( y, X, groups )
%crossValidationRegression

trRMSE = [];
teRMSE = [];

for groupNumber = 1:groups
    %% Separe X and y in train and test parts
    [ XTr, XTe, yTr, yTe ] = divideDataSet( X, y, groups, groupNumber );
    
    %% Make prediction for test data
    yTrPred = predict( XTr, yTr, XTr );
    yTePred = predict( XTr, yTr, XTe );
    
    trRMSE = [trRMSE computeRMSE(yTr, yTrPred)];
    teRMSE = [teRMSE computeRMSE(yTe, yTePred)];
    
end

end

