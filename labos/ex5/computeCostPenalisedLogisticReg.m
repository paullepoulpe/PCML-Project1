function logLikelihood = computeCostPenalisedLogisticReg(y, tX, beta, lambda)

logLikelihood = - sum( y.*(tX*beta) - log(ones(length(y),1) + exp(tX*beta))) ...
    +lambda*sum(beta.^2);

end