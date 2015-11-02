function hessian = computeHessian(tX, beta)

S = computeSigma(tX*beta).*(ones(size(tX,1),1)-computeSigma(tX*beta));
hessian = tX'*diag(S)*tX;

end