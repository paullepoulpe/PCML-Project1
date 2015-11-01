%% Clean workspace & Load data
clear
clc

addpath('../utils');
addpath('../methods');

load('../../data/SaoPaulo_regression.mat');
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

%% Strong correlation
section('Strong correlation');

[I, corrs] = correlation(X_train, y_train);

% Normalize
corrs = normalise(corrs);

% Find outliers
strongCorrIdx = find(corrs > 2);

fprintf('Dimensions with strong correlation to y:');
disp(I(strongCorrIdx));


