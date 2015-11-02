function gradient = computeGradient(y, tX, beta)

gradient = tX'*(computeSigma(tX*beta)-y);

end