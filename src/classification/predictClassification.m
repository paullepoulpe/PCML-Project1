function [ groups ] = predictClassification( tX, beta, k )
%predictClass Return the most probable class for each data


for class = 1:k
    
    yPred(:,class) = sigma(tX * beta(:,class));
    
end

[~,I] = max(yPred');
groups = I';

end

