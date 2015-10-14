function logLikelihood = computeCostLogisticReg(y, tX, beta)

logLikelihood = - sum( y.*(tX*beta) - ln(ones(length(y),1) + exp(tX*beta)));

end