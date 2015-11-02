function [ XTr, XTe, yTr, yTe ] = divideDataSet( X, y, K, k )
%divideDataSet Divide the dataset in train and test parts

N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);

for k = 1:K
    idxCV(k, :) = idx(1+(k-1)*Nk:k*Nk);
end

% Get k'th subgroup in test, others in train
idxTe = idxCV(k, :);
idxTr = idxCV([1:k-1 k+1:end], :);
idxTr = idxTr(:);
yTe = y(idxTe);
XTe = X(idxTe, :);
yTr = y(idxTr);
XTr = X(idxTr, :);

end

