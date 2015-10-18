function [L, g, H] = logisticRegLoss(beta, y, tX)

L = computeCostLogisticReg(y, tX, beta);
g = computeGradient(y, tX, beta);
H = computeHessian(tX, beta);

end