close all
clear
clc

load('../data/SaoPaulo_regression.mat')

%% Visualise the data
figure()
histogram(y_train);

figure()
plot(X_train(:,55),y_train,'*');
% About X data:
% Binary dimensions :50
% Three values dimensions : 20, 21, 26, 37, 46
% Four values dimensions : 32, 45, 51, 65
% Two clouds : 55

% refactor to use find
binaryDim = [50];
threeDim = [20 21 26 37 46];
fourDim = [32 45 51 65];
twoCloudsDim = [6 55]; 

%% Cross-validation
datagroups = 3;
group = 2;
% for group = 1:datagroups
    
    %% Separe X_train in train and test to make cross-validation
    [ XTr, XTe, yTr, yTe ] = divideDataSet( X_train, y_train, datagroups, group );
    
    %% Clustering of the data with kMeans
    % Need to do that with non-normalised data !
    X = XTr;
    y = yTr;
    [cluster, centers] = kmeans([X(:,55) y], 2);
    [subcluster, subcenters] = kmeans([X(cluster==1,6) y(cluster==1)], 2);
    
    finalcluster = cluster;
    finalcluster(cluster == 2) = 3;
    finalcluster(cluster==1) = subcluster;
    finalcenters = [subcenters(1,:);centers(2,:);subcenters(2,:)];
    
    figure()
    plot(X(cluster==1,55),y(cluster==1,1),'*');
    hold on
    plot(X(cluster==2,55),y(cluster==2,1),'*');
    plot(XTe(:,55),yTe(:,1),'*');
    plot(centers(:,1), centers(:,end),'.', 'markersize',30);
    plot(subcenters(:,1), subcenters(:,end),'.', 'markersize',30);
    plot(finalcenters(:,1), finalcenters(:,end),'.r', 'markersize',30);
    legend('cluster 1','cluster 2','test data','centers','subcenters','finalcenters');
    
    figure()
    plot(X(finalcluster==1,55),y(finalcluster==1,1),'*');
    hold on
    plot(X(finalcluster==2,55),y(finalcluster==2,1),'*');
    plot(X(finalcluster==3,55),y(finalcluster==3,1),'*');
    plot(XTe(:,55),yTe(:,1),'*');
    plot(finalcenters(:,1), finalcenters(:,end),'.r', 'markersize',30);
    legend('cluster 1','cluster 2','cluster 3','test data','finalcenters');
    
    figure()
    plot3(X(finalcluster==1,6),X(finalcluster==1,55),y(finalcluster==1,1),'*');
    hold on
    plot3(X(finalcluster==2,6),X(finalcluster==2,55),y(finalcluster==2,1),'*');
    plot3(X(finalcluster==3,6),X(finalcluster==3,55),y(finalcluster==3,1),'*');
    plot3(XTe(:,6),XTe(:,55),yTe(:,1),'*');
%     plot3(finalcenters(:,1),centers(:,2), finalcenters(:,end),'.r', 'markersize',30);
    legend('cluster 1','cluster 2','cluster 3','test data','finalcenters');
    %% Normalise the data (train)
    
    [XTrNormalised, meanX, stdX] = normalise(XTr);
    [yTrNormalised, meanY, stdY] = normalise(yTr);
    
    %% Remove outliers (train)
    
    %[XFiltered, yFiltered] = removeOutlierLines(XNormalised, yNormalised, 3, 2);
    XTrFiltered = fixOutliers(XTrNormalised, 3);
    % XFiltered = XNormalised;
    yTrFiltered = yTrNormalised;
    
    len = length(XTrFiltered);
    width = size(XTrFiltered, 2);
    
    %% Remove correlated columns using PCA (test)
    [XKept, V, VReduced] = pca(XTrFiltered, 1);
    
    %% Preprocess test data
    % Find cluster
    clusterTrue = findClosestGroup([XTe(:,55) yTe], finalcenters);
    
    % Normalise given mean and std of the training set
    XTeNormalised = (XTe - ones(size(XTe,1),1)*meanX)./(ones(size(XTe,1),1)*stdX);
    yTeNormalised = (yTe - meanY)./stdY;
    % Reduce the dimension given PCA of training set
    XTeNormalised = (VReduced'*(V*XTeNormalised'))';
    
    %% Train for clusters
    
    tXTr = [ones(length(XKept), 1)  XKept];
    tXTe = [ones(length(XTeNormalised), 1)  XTeNormalised];
    
%% Using minimum distance to the center in 55th dimension
    clusterPred = findClosestGroup([XTe(:,55)], centers(:,1));
    figure()
    plot(XTe(clusterPred==1,55),yTe(clusterPred==1,1),'*');
    hold on
    plot(XTe(clusterPred==2,55),yTe(clusterPred==2,1),'*');
    finalclusterPred = clusterPred;
    finalclusterPred(clusterPred == 2) = 3;
    
    beta = leastSquaresGD(y(finalcluster~=3), tXTr(finalcluster~=3,:), 0.01);
    yPred = (tXTe(finalclusterPred~=3,:) * beta)*stdY + meanY;
%     betaCl = logisticRegression( cluster(cluster~=3), [XTr(cluster~=3,55) yTr(cluster~=3)], 0.001 );
%     subclusterPred =  sigma([XTe(finalclusterPred~=3,55) yPred]* betaCl);
%     subclusterPred(subclusterPred > 0.5) = 1;
%     subclusterPred(subclusterPred <= 0.5) = 2;
    subclusterPred = findClosestGroup([XTe(finalclusterPred~=3,6) ], finalcenters(1:2,1));
    finalclusterPred(finalclusterPred~=3) = subclusterPred;
    
    figure()
    plot(XTe(finalclusterPred==1,55),yTe(finalclusterPred==1,1),'*');
    hold on
    plot(XTe(finalclusterPred==2,55),yTe(finalclusterPred==2,1),'*');
    plot(XTe(finalclusterPred==3,55),yTe(finalclusterPred==3,1),'*');
%     plot(finalcenters(:,1), finalcenters(:,end),'.r', 'markersize',30);
    legend('cluster 1','cluster 2','cluster 3');
    %% Train for clusters
    %% With an approximation of y and minimum distance to centers
    % y = yFiltered;
    % % beta = logisticRegression(y, tX, 0.01);
    % % beta = penLogisticRegression(y, tX, 0.01,1000);
    % % beta = leastSquaresGD(y, tX, 0.01);
    % betaCl = leastSquares(y, tXTr);
    % % beta = ridgeRegression(y, tX, 10);
    %
    % yPred = (tXTe * betaCl)*stdY + meanY;
    % clusterPred = findClosestGroup([XTe yPred], centers);
    %% With logistic regression for multiclass
%     y = cluster;
%     betaC = logRegMultiClass( y, tXTr, 0.001, 4 );
%     clusterPred = predictClass(tXTe, betaC, 4);
    
    figure()
    plot(clusterTrue, finalclusterPred,'.','markersize',10);
    axis([0 5 0 5]);
    
    %% Train
    y = yTrFiltered;
    
    for k = 1:3
        % beta = leastSquaresGD(y(cluster == k), tXTr(cluster == k,:), 0.01);
        beta(:,k) = leastSquares(y(finalcluster == k), tXTr(finalcluster == k,:));
        % beta = ridgeRegression(y(cluster == k), tXTr(cluster == k,:), 10);
        
    end
    
    yPred = zeros(size(yTe));
    for k = 1:3
        indices = find(finalclusterPred == k);
        for i = 1:length(indices)
            yPred(indices(i),:) = tXTe(i,:) * beta(:,k);
        end
    end
    
%     betaT = leastSquares(yTr, tXTr);
%     yPred = tXTe*betaT;
    yPred = yPred * stdY + meanY;
    yTrue = yTe;
    
    figure()
    plot(yTrue, yPred, '*');
    hold on;
    maxY = ceil(max([yPred yTrue]));
    minY = floor(min([yPred yTrue]));
    plot(minY:maxY, minY:maxY);
    
% end