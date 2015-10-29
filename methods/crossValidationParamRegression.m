function [ trRMSE, teRMSE, param ] = crossValidationParamRegression( y, X, groups )
%crossValidationRegression

trRMSE = [];
teRMSE = [];

param = logspace(-5,8,20);

for paramNumber = 1:length(param)
    trError = [];
    teError = [];
    lambda = param(paramNumber);
    for groupNumber = 1:groups
        %% Separe X and y in train and test parts
        [ XTr, XTe, yTr, yTe ] = divideDataSet( X, y, groups, groupNumber );
        
        %% Make prediction for test data
        yTrPred = predict( XTr, yTr, XTr, lambda );
        yTePred = predict( XTr, yTr, XTe, lambda );
        
        trError = [trError computeRMSE(yTr, yTrPred)];
        teError = [teError computeRMSE(yTe, yTePred)];
    end
    trRMSE = [trRMSE mean(trError)];
    teRMSE = [teRMSE mean(teError)];
    
end

end

