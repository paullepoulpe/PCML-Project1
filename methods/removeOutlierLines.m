function [XFiltered, yFiltered] = removeOutlierLines(X, y, limitDeviation, limitPerLine)
%removeOutlierLines Removes lines that contain too many outliers

stdX = std(X);
meanX = mean(X);
len = length(X);

% Number of elements that are further than limitDeviation * std deviations away per line
counts = zeros(1, len);
for i = 1:len
    normalisedLine = (X(i,:) - meanX) ./ stdX;
    counts(i) = sum(abs(normalisedLine) > limitDeviation);
end

% Remove all line that have 2 or more outliers
XFiltered = X(counts < limitPerLine, :);
yFiltered = y(counts < limitPerLine, :);

end

