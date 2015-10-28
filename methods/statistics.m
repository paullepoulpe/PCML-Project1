%% Clean workspace & Load data
clear
clc

load('../data/SaoPaulo_regression.mat');
len = length(X_train);
dim = size(X_train, 2);

section = @(text) fprintf('\n\n%s\n%s\n', text, repmat('=', 1, length(text)));


%% Cardinalities
section('Cardinalities');

cards = cardinalities(X_train);

for card = unique(cards)
    dims = find(cards == card);
    fprintf('%2d dimensions with cardinality %d: ', length(dims), card);
    if(card < len)
        disp(dims);
    else
        fprintf('All other dimesions\n');
    end
end

%% Clusters
section('2-way Clusters');

sums = zeros(1, dim);
for d = 1:dim
   column = X_train(:, d);
   [idx, C, sumd] = kmeansStable(column, 2);
   sums(d) = norm(sumd);
end

% Normalize
sums = normalise(sums);

% Find outliers ????
dimsClustered = find(sums > 2);

fprintf('Dimensions with well defined 2-way clusters:');
disp(dimsClustered);


