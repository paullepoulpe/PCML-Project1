function logLikelihood = computeCostLogisticReg(y, tX, beta)

logLikelihood = - sum( y.*(tX*beta) - log(ones(length(y),1) + exp(tX*beta)));

end