function [ yPred, pPred ] = predictClassification( XTr, yTr, XTe, param )
%predictClassification Prediction function for the classification

%% Separe data set in function of the 20th dimension of X
cluster = XTr(:,20);
clusterPred = XTe(:,20);

%% Find cardinalities, remove binary dimensions
cards = cardinalities( XTr );
XTr = XTr(:,cards > 2);
XTe = XTe(:,cards > 2);

%% Train classification model for each cluster
sizeCluster = length(unique(cluster));
tXTr = cell(sizeCluster,1);
yTrF = cell(sizeCluster,1);
beta = cell(sizeCluster,1);

for cl = 1:sizeCluster
    % Take the data for one cluster
    X = XTr(cluster == cl,:);
    y = yTr(cluster == cl,:);
    
    % Transform y to have only 1 and 0 
    y(y < 0) = 0;
    
    % Separate discrete and continuous dimensions
    cardsX(cl,:) = cardinalities( X );
    disX = X(:,cardsX(cl,:) ~= size(X,1));
    contX = X(:,cardsX(cl,:) == size(X,1));
    
    % Normalise the continuous data
    [XTrNormalised, meanX(cl,:), stdX(cl,:)] = normalise(contX);
    
    % Remove outliers
    [XTrFiltered, yTrFiltered, linesKept] = removeOutlierLines(XTrNormalised, y, 3, 1);
    disX = disX(linesKept == 1,:);
    
    % Put discrete and continuous dimensions together again
    XTrKept = [XTrFiltered, disX];
    
    % Compute tX and final y
    tXTr{cl,1} = [ones(length(XTrKept), 1)  XTrKept];
    yTrF{cl,1} = yTrFiltered;
    
    % Compute (penalised) logistic regression
%     beta{cl,1} = logisticRegression(yTrF{cl,1}, tXTr{cl,1}, 0.001);
    beta{cl,1} = penLogisticRegression(yTrF{cl,1}, tXTr{cl,1}, 0.001, 0.01);

end

%% Test
pPred = zeros(size(XTe,1),1);
yPred = zeros(size(XTe,1),1);
for cl = 1:length(unique(clusterPred))
    % Take the data from one cluster
    X = XTe(clusterPred == cl,:);
    disX = X(:,cardsX(cl,:) < 10);
    contX = X(:,cardsX(cl,:) >= 10);
    
    % Normalise given mean and std of the training set
    XTeNormalised = (contX - ones(size(contX,1),1)*meanX(cl,:))./(ones(size(contX,1),1)*stdX(cl,:));
    
    XTeKept = [XTeNormalised, disX];
    
    % Compute tX
    tXTe = [ones(length(XTeNormalised), 1)  XTeKept];
    
    % Probability prediction
    pPred(clusterPred == cl,:) = sigma(tXTe * beta{cl,1});

end

% Prediction given probability
yPred(pPred >= 0.5) = 1;
yPred(pPred < 0.5) = -1;

end

