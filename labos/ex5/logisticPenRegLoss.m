function [L, g, H] = logisticPenRegLoss(beta, y, tX, lambda)

L = computeCostPenalisedLogisticReg(y, tX, beta, lambda);
g = computeGradient(y, tX, beta);
H = computeHessian(tX, beta);

end