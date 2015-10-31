function [ beta ] = logRegMultiClass( y, tX, alpha, k )
%logRegMultiClass Makes logistic regression for k classes

for class = 1:k
    
    yClass = y;
    yClass(y == class,:) = 1;
    yClass(y ~= class,:) = 0;
    beta(:,class) = logisticRegression(yClass, tX, alpha);
    
end

end

