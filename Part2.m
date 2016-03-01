%% Load data

load('DataF');

%% Normalization

[r,c] = size(fea);

for i=1:c
    meanval = mean(fea(:,i));
    
    for j=1:r
        fea(j,i) = fea(j,i) - meanval;
    end
end

%% Apply PCA to reduce the dimension to be 4

% covMatrix = cov(fea);
% [eigenVector, eigenValues] = eig(covMatrix);
% feaD4 = fea*eigenVector(:, end:-1:end-3);

[~,score,latent,~,~,~] = pca(fea);
feaD4 = score(:,1:4);

%% Cluster the data using agglomerative algorithms in hierarchical clustering 

%Single Link
% tree_singlelink = linkage(feaD4, 'single');
% cluster_singlelink = cluster(tree_singlelink, 'maxclust', 10);
cluster_singlelink = clusterdata(feaD4, 'linkage', 'single', 'maxclust', 10);


%evaluate the clustering result in terms of Separation-Index, Rand-Index, and F-measure
%???cluster index in [1,10], class label [0,9]
error_singlelink =  sum(cluster_singlelink ~= gnd);
%???how to caculate tp,fp,fn in multi-class
% tpfp_singlelink = sum(cluster_singlelink == 1);
% tpfn_singlelink = sum(gnd == 1);
% tp_singlelink = sum(cluster_singlelink == (gnd == 1));



Accuracy_singlelink = (1-error_singlelink) / r *100;
Precision_singlelink = tp_singlelink/tpfp_singlelink;
Recall_singlelink = tp_singlelink/tpfn_singlelink;
Fmeasure_singlelink = 2 / (1/Precision_singlelink + 1/Recall_singlelink);

%Complete Link
cluster_completelink = clusterdata(feaD4, 'linkage', 'complete', 'maxclust', 10);

%Ward's Algorithm(minimum variance algorithm)
cluster_ward = clusterdata(feaD4, 'linkage', 'ward', 'maxclust', 10);
%the number of clusters from 2 to 15
for i = 2:15
    cluster_ward = clusterdata(feaD4, 'linkage', 'ward', 'maxclust', i);
end

%% Cluster the data using K-means algorithm
%the number of clusters from 2 to 15
for i = 2:15
    [cluster_kmeans,centroid] = kmeans(feaD4, i);
end
%% Cluster the data using Fuzzy C-means algorithm
%set the exponent for partition matrix to 2
options = [2; nan; nan ; nan];
[center, cluster_fcmeans, obj_fcn] = fcm(feaD4, 10, options);
